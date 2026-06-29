import base64
import json

import cv2
import numpy as np
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db import IntegrityError, transaction
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .facenet_utils import cosine_similarity, estimate_image_quality, get_face_embedding
from .forms import CandidateForm, ElectionForm, LoginForm, SignUpForm
from .iris_utils import detect_passive_liveness, extract_iris_features, iris_distance
from .models import (
    AuditLog,
    Candidate,
    Election,
    FaceBiometricSample,
    IrisBiometricSample,
    VoteRecord,
    Voter,
)


FACE_THRESHOLD = 0.75
IRIS_THRESHOLD = 0.20
MIN_FACE_QUALITY = 0.35
MIN_IRIS_QUALITY = 0.15


def get_client_ip(request):
    forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '')


def log_audit(request, action, status='info', voter=None, election=None, details=None):
    AuditLog.objects.create(
        action=action,
        status=status,
        user=request.user if request.user.is_authenticated else None,
        voter=voter,
        election=election,
        ip_address=get_client_ip(request),
        user_agent=request.META.get('HTTP_USER_AGENT', '')[:255],
        details=details or {},
    )


def get_active_election():
    now = timezone.now()
    election = (
        Election.objects.filter(status=Election.STATUS_ACTIVE, starts_at__lte=now, ends_at__gte=now)
        .prefetch_related('candidates')
        .first()
    )
    if election:
        return election
    return Election.objects.filter(status=Election.STATUS_ACTIVE).prefetch_related('candidates').first()


@login_required
def home(request):
    active_election = get_active_election()
    total_voters = Voter.objects.filter(is_active=True, is_approved=True).count()
    total_votes = VoteRecord.objects.count()
    results_preview = []

    if active_election:
        results_preview = list(
            active_election.candidates.annotate(total_votes=Count('vote_records')).order_by('-total_votes', 'display_order')
        )

    return render(
        request,
        'voting/index.html',
        {
            'username': request.user.username,
            'active_election': active_election,
            'total_voters': total_voters,
            'total_votes': total_votes,
            'turnout_percent': round((total_votes / total_voters) * 100, 2) if total_voters else 0,
            'results_preview': results_preview,
        },
    )


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    form = LoginForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        log_audit(request, 'auth_login', status='success', details={'username': user.username})
        return redirect('home')

    return render(request, 'voting/login.html', {'form': form})


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    form = SignUpForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Account created successfully. Please log in.')
        return redirect('login')

    return render(request, 'voting/signup.html', {'form': form})


def logout_view(request):
    if request.user.is_authenticated:
        log_audit(request, 'auth_logout', status='info', details={'username': request.user.username})
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')


def decode_request_image(request):
    uploaded_image = request.FILES.get('image')
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    captured_image = request.POST.get('captured_image')
    if not captured_image:
        return None

    try:
        _, encoded = captured_image.split(',', 1)
        file_bytes = base64.b64decode(encoded)
    except (ValueError, TypeError, base64.binascii.Error):
        return None

    return cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def extract_biometrics(frame):
    if frame is None:
        return None, 'Could not read the camera image.'

    face_embedding = get_face_embedding(frame)
    if face_embedding is None:
        return None, 'Face not detected. Please look straight into the camera.'

    iris_features = extract_iris_features(frame)
    if iris_features is None:
        return None, 'Iris not detected properly. Please move closer and keep your eyes visible.'

    liveness_ok, liveness_message = detect_passive_liveness(iris_features)
    if not liveness_ok:
        return None, liveness_message

    image_quality = estimate_image_quality(frame)
    if image_quality['quality_score'] < MIN_FACE_QUALITY:
        return None, 'Image quality is too low. Improve lighting and keep the camera steady.'

    if iris_features['quality_score'] < MIN_IRIS_QUALITY:
        return None, 'Eye visibility is too weak for reliable iris verification. Please try again.'

    return {
        'face_embedding': face_embedding,
        'face_quality': image_quality,
        'iris_features': iris_features,
    }, None


@login_required
def register_voter(request):
    if request.method == 'POST':
        name = (request.POST.get('name') or '').strip()
        email = (request.POST.get('email') or '').strip()
        voter_id = (request.POST.get('voter_id') or '').strip()
        frame = decode_request_image(request)

        log_audit(request, 'register_attempt', status='info', details={'voter_id': voter_id})

        if not name or not voter_id:
            return render(request, 'voting/result.html', {
                'message': 'Name and voter ID are required.',
                'details': 'Please provide both fields before registration.',
            })

        biometrics, error_message = extract_biometrics(frame)
        if error_message:
            log_audit(request, 'register_attempt', status='failure', details={'voter_id': voter_id, 'error': error_message})
            return render(request, 'voting/result.html', {'message': error_message})

        try:
            with transaction.atomic():
                voter = Voter.objects.create(
                    name=name,
                    email=email,
                    voter_id=voter_id,
                    is_active=True,
                    is_approved=True,
                )
                FaceBiometricSample.objects.create(
                    voter=voter,
                    embedding=json.dumps(biometrics['face_embedding']),
                    quality_score=biometrics['face_quality']['quality_score'],
                    brightness_score=biometrics['face_quality']['brightness'],
                    blur_score=biometrics['face_quality']['blur_score'],
                    is_reference=True,
                )
                IrisBiometricSample.objects.create(
                    voter=voter,
                    left_features=json.dumps(biometrics['iris_features']['left']),
                    right_features=json.dumps(biometrics['iris_features']['right']),
                    quality_score=biometrics['iris_features']['quality_score'],
                    is_reference=True,
                )
        except IntegrityError:
            log_audit(request, 'register_attempt', status='failure', details={'voter_id': voter_id, 'error': 'duplicate_voter_id'})
            return render(request, 'voting/result.html', {
                'message': 'Voter ID already registered',
                'details': f'Voter ID {voter_id} is already in the system.',
            })

        log_audit(request, 'register_success', status='success', voter=voter, details={'voter_id': voter_id})
        return render(request, 'voting/result.html', {
            'message': 'Voter Registered Successfully',
            'details': (
                f'Name: {voter.name} | Voter ID: {voter.voter_id} | '
                f'Face quality: {biometrics["face_quality"]["quality_score"]:.2f} | '
                f'Iris quality: {biometrics["iris_features"]["quality_score"]:.2f}'
            ),
        })

    return render(request, 'voting/register.html', {'active_election': get_active_election()})


def find_matching_voter(live_face, live_iris):
    best_match = None
    best_face_score = -1.0
    best_iris_score = float('inf')

    voters = (
        Voter.objects.filter(is_active=True, is_approved=True)
        .prefetch_related('face_samples', 'iris_samples')
    )

    for voter in voters:
        face_samples = list(voter.face_samples.all())
        iris_samples = list(voter.iris_samples.all())
        if not face_samples or not iris_samples:
            continue

        for face_sample in face_samples:
            stored_face = json.loads(face_sample.embedding)
            face_score = cosine_similarity(live_face, stored_face)
            if face_score <= FACE_THRESHOLD:
                continue

            for iris_sample in iris_samples:
                stored_iris = {
                    'left': json.loads(iris_sample.left_features),
                    'right': json.loads(iris_sample.right_features),
                }
                iris_score = iris_distance(live_iris, stored_iris)
                if iris_score < IRIS_THRESHOLD:
                    if face_score > best_face_score or (face_score == best_face_score and iris_score < best_iris_score):
                        best_match = voter
                        best_face_score = face_score
                        best_iris_score = iris_score

    return best_match, best_face_score, best_iris_score


@login_required
def vote(request):
    active_election = get_active_election()
    candidates = active_election.candidates.all() if active_election else Candidate.objects.none()

    if request.method == 'POST':
        if not active_election or not active_election.is_open:
            return render(request, 'voting/result.html', {
                'message': 'No active election is open right now.',
                'details': 'Please contact an election officer or return when the election window opens.',
            })

        candidate = get_object_or_404(Candidate, pk=request.POST.get('candidate_id'), election=active_election)
        frame = decode_request_image(request)
        biometrics, error_message = extract_biometrics(frame)
        log_audit(request, 'vote_attempt', status='info', election=active_election, details={'candidate_id': candidate.pk})

        if error_message:
            log_audit(request, 'vote_rejected', status='failure', election=active_election, details={'error': error_message})
            return render(request, 'voting/result.html', {'message': error_message})

        voter, face_score, iris_score = find_matching_voter(
            biometrics['face_embedding'],
            {
                'left': biometrics['iris_features']['left'],
                'right': biometrics['iris_features']['right'],
            },
        )

        if voter is None:
            log_audit(request, 'vote_rejected', status='failure', election=active_election, details={'error': 'biometric_mismatch'})
            return render(request, 'voting/result.html', {
                'message': 'Voter Not Found or Biometric Mismatch',
                'details': 'The detected face and iris do not match any registered voter.',
            })

        if VoteRecord.objects.filter(voter=voter, election=active_election).exists():
            log_audit(request, 'vote_rejected', status='failure', voter=voter, election=active_election, details={'error': 'already_voted'})
            return render(request, 'voting/result.html', {
                'message': 'Vote Already Completed',
                'details': f'Name: {voter.name} | Voter ID: {voter.voter_id}',
            })

        with transaction.atomic():
            VoteRecord.objects.create(voter=voter, election=active_election, candidate=candidate)
            voter.last_verified_at = timezone.now()
            voter.save(update_fields=['last_verified_at'])

        log_audit(
            request,
            'vote_success',
            status='success',
            voter=voter,
            election=active_election,
            details={
                'candidate': candidate.name,
                'face_score': round(face_score, 4),
                'iris_score': round(iris_score, 4),
            },
        )

        return render(request, 'voting/result.html', {
            'message': 'Vote Success',
            'details': (
                f'Election: {active_election.name} | Candidate: {candidate.name} | '
                f'Voter: {voter.name} ({voter.voter_id}) | '
                f'Face matched: {face_score:.2f} | Iris matched: {iris_score:.3f}'
            ),
        })

    return render(request, 'voting/vote.html', {
        'active_election': active_election,
        'candidates': candidates,
    })


@login_required
def results(request):
    selected_election = get_active_election() or Election.objects.order_by('-starts_at').first()
    election_id = request.GET.get('election')
    if election_id:
        selected_election = get_object_or_404(Election, pk=election_id)

    result_rows = []
    total_votes = 0
    if selected_election:
        result_rows = list(
            selected_election.candidates.annotate(total_votes=Count('vote_records')).order_by('-total_votes', 'display_order')
        )
        total_votes = sum(row.total_votes for row in result_rows)

    return render(request, 'voting/results.html', {
        'selected_election': selected_election,
        'elections': Election.objects.all(),
        'result_rows': result_rows,
        'total_votes': total_votes,
    })


@login_required
@user_passes_test(lambda user: user.is_superuser)
def superuser_panel(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'create_election':
            election_form = ElectionForm(request.POST)
            candidate_form = CandidateForm()
            if election_form.is_valid():
                election = election_form.save()
                log_audit(
                    request,
                    'election_create',
                    status='success',
                    election=election,
                    details={'message': 'Election created from superuser panel'},
                )
                messages.success(request, f'Election "{election.name}" created successfully.')
                return redirect('superuser_panel')
        elif action == 'create_candidate':
            election_form = ElectionForm()
            candidate_form = CandidateForm(request.POST)
            if candidate_form.is_valid():
                candidate = candidate_form.save()
                log_audit(
                    request,
                    'election_create',
                    status='info',
                    election=candidate.election,
                    details={'message': 'Candidate created from superuser panel', 'candidate': candidate.name},
                )
                messages.success(request, f'Candidate "{candidate.name}" added successfully.')
                return redirect('superuser_panel')
        elif action == 'delete_voter':
            election_form = ElectionForm()
            candidate_form = CandidateForm()
            voter = get_object_or_404(Voter, pk=request.POST.get('voter_id'))
            voter_name = voter.name
            voter_identifier = voter.voter_id
            log_audit(
                request,
                'voter_delete',
                status='success',
                details={'name': voter_name, 'voter_id': voter_identifier},
            )
            voter.delete()
            messages.success(request, f'Voter "{voter_name}" deleted successfully.')
            return redirect('superuser_panel')
        elif action == 'delete_candidate':
            election_form = ElectionForm()
            candidate_form = CandidateForm()
            candidate = get_object_or_404(Candidate, pk=request.POST.get('candidate_id'))
            candidate_name = candidate.name
            election = candidate.election
            candidate.delete()
            log_audit(
                request,
                'election_create',
                status='info',
                election=election,
                details={'message': 'Candidate deleted from superuser panel', 'candidate': candidate_name},
            )
            messages.success(request, f'Candidate "{candidate_name}" deleted successfully.')
            return redirect('superuser_panel')
        else:
            election_form = ElectionForm()
            candidate_form = CandidateForm()
    else:
        election_form = ElectionForm()
        candidate_form = CandidateForm()

    elections = Election.objects.annotate(candidate_total=Count('candidates'), vote_total=Count('vote_records')).order_by('-starts_at')
    candidate_list = Candidate.objects.select_related('election').order_by('election__name', 'display_order', 'name')[:100]
    recent_voters = (
        Voter.objects
        .prefetch_related('face_samples', 'iris_samples', 'vote_records__election', 'vote_records__candidate')
        .order_by('-registered_at')[:8]
    )
    recent_logs = AuditLog.objects.select_related('user', 'voter', 'election').order_by('-created_at')[:8]
    voter_histories = []
    for voter in recent_voters:
        vote_records = list(voter.vote_records.all())
        voter_histories.append({
            'id': voter.id,
            'name': voter.name,
            'voter_id': voter.voter_id,
            'email': voter.email,
            'registered_at': voter.registered_at,
            'last_verified_at': voter.last_verified_at,
            'face_sample_count': len(voter.face_samples.all()),
            'iris_sample_count': len(voter.iris_samples.all()),
            'vote_count': len(vote_records),
            'votes': vote_records[:3],
        })

    return render(request, 'voting/superuser_panel.html', {
        'active_election': get_active_election(),
        'total_elections': Election.objects.count(),
        'active_elections': Election.objects.filter(status=Election.STATUS_ACTIVE).count(),
        'total_candidates': Candidate.objects.count(),
        'total_voters': Voter.objects.count(),
        'approved_voters': Voter.objects.filter(is_approved=True).count(),
        'total_votes': VoteRecord.objects.count(),
        'audit_events': AuditLog.objects.count(),
        'elections': elections[:6],
        'candidate_list': candidate_list,
        'voter_histories': voter_histories,
        'recent_logs': recent_logs,
        'election_form': election_form,
        'candidate_form': candidate_form,
    })
