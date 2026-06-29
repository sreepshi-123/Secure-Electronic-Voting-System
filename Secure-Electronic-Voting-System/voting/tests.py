from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .models import Candidate, Election, VoteRecord, Voter


class VotingFlowTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='pass12345')
        self.client.login(username='tester', password='pass12345')
        now = timezone.now()
        self.election = Election.objects.create(
            name='Student Council 2026',
            description='Campus-wide election',
            starts_at=now - timezone.timedelta(hours=1),
            ends_at=now + timezone.timedelta(hours=1),
            status=Election.STATUS_ACTIVE,
        )
        self.candidate = Candidate.objects.create(
            election=self.election,
            name='Asha Mehta',
            party='Forward Campus',
            display_order=1,
        )
        self.biometric_payload = {
            'face_embedding': [0.1, 0.2, 0.3],
            'face_quality': {'quality_score': 0.8, 'brightness': 0.5, 'blur_score': 120.0},
            'iris_features': {
                'left': [0.1, 0.2, 0.3, 0.4],
                'right': [0.1, 0.2, 0.3, 0.4],
                'quality_score': 0.7,
            },
        }

    @patch('voting.views.extract_biometrics')
    @patch('voting.views.decode_request_image')
    def test_register_voter_creates_biometric_samples(self, mock_decode, mock_extract):
        mock_decode.return_value = object()
        mock_extract.return_value = (self.biometric_payload, None)

        response = self.client.post(reverse('register'), {
            'name': 'Asha Mehta',
            'email': 'asha@example.com',
            'voter_id': 'V1001',
            'captured_image': 'data:image/jpeg;base64,dummy',
        })

        self.assertContains(response, 'Voter Registered Successfully')
        voter = Voter.objects.get(voter_id='V1001')
        self.assertEqual(voter.face_samples.count(), 1)
        self.assertEqual(voter.iris_samples.count(), 1)

    @patch('voting.views.find_matching_voter')
    @patch('voting.views.extract_biometrics')
    @patch('voting.views.decode_request_image')
    def test_vote_creates_vote_record(self, mock_decode, mock_extract, mock_match):
        voter = Voter.objects.create(name='Asha Mehta', voter_id='V1001', email='asha@example.com')
        mock_decode.return_value = object()
        mock_extract.return_value = (self.biometric_payload, None)
        mock_match.return_value = (voter, 0.91, 0.08)

        response = self.client.post(reverse('vote'), {
            'candidate_id': self.candidate.id,
            'captured_image': 'data:image/jpeg;base64,dummy',
        })

        self.assertContains(response, 'Vote Success')
        self.assertTrue(VoteRecord.objects.filter(voter=voter, election=self.election, candidate=self.candidate).exists())

    @patch('voting.views.find_matching_voter')
    @patch('voting.views.extract_biometrics')
    @patch('voting.views.decode_request_image')
    def test_repeated_vote_is_blocked(self, mock_decode, mock_extract, mock_match):
        voter = Voter.objects.create(name='Asha Mehta', voter_id='V1001', email='asha@example.com')
        VoteRecord.objects.create(voter=voter, election=self.election, candidate=self.candidate)
        mock_decode.return_value = object()
        mock_extract.return_value = (self.biometric_payload, None)
        mock_match.return_value = (voter, 0.91, 0.08)

        response = self.client.post(reverse('vote'), {
            'candidate_id': self.candidate.id,
            'captured_image': 'data:image/jpeg;base64,dummy',
        })

        self.assertContains(response, 'Vote Already Completed')

    @patch('voting.views.extract_biometrics')
    @patch('voting.views.decode_request_image')
    def test_closed_election_cannot_accept_vote(self, mock_decode, mock_extract):
        self.election.ends_at = timezone.now() - timezone.timedelta(minutes=1)
        self.election.save()
        mock_decode.return_value = object()
        mock_extract.return_value = (self.biometric_payload, None)

        response = self.client.post(reverse('vote'), {
            'candidate_id': self.candidate.id,
            'captured_image': 'data:image/jpeg;base64,dummy',
        })

        self.assertContains(response, 'No active election is open right now.')

    def test_results_page_shows_candidate_totals(self):
        voter = Voter.objects.create(name='Asha Mehta', voter_id='V1001')
        VoteRecord.objects.create(voter=voter, election=self.election, candidate=self.candidate)

        response = self.client.get(reverse('results'))

        self.assertContains(response, 'Election Results')
        self.assertContains(response, 'Asha Mehta')
        self.assertContains(response, '1 vote')

    def test_superuser_panel_requires_superuser(self):
        response = self.client.get(reverse('superuser_panel'))
        self.assertEqual(response.status_code, 302)

    def test_superuser_panel_loads_for_superuser(self):
        admin_client = Client()
        admin_user = User.objects.create_superuser(username='admin', email='admin@example.com', password='pass12345')
        admin_client.login(username='admin', password='pass12345')

        response = admin_client.get(reverse('superuser_panel'))

        self.assertContains(response, 'Superuser Control Room')
        self.assertContains(response, admin_user.username, status_code=200)

    def test_superuser_can_create_election_from_ui(self):
        admin_client = Client()
        User.objects.create_superuser(username='admin', email='admin@example.com', password='pass12345')
        admin_client.login(username='admin', password='pass12345')

        response = admin_client.post(reverse('superuser_panel'), {
            'action': 'create_election',
            'name': 'City Election 2026',
            'description': 'Created from UI',
            'starts_at': '2026-04-16T10:00',
            'ends_at': '2026-04-16T18:00',
            'status': Election.STATUS_DRAFT,
        }, follow=True)

        self.assertContains(response, 'created successfully', status_code=200)
        self.assertTrue(Election.objects.filter(name='City Election 2026').exists())

    def test_superuser_can_delete_voter_from_ui(self):
        admin_client = Client()
        User.objects.create_superuser(username='admin', email='admin@example.com', password='pass12345')
        admin_client.login(username='admin', password='pass12345')
        voter = Voter.objects.create(name='Delete Me', voter_id='DEL-01', email='delete@example.com')

        response = admin_client.post(reverse('superuser_panel'), {
            'action': 'delete_voter',
            'voter_id': voter.id,
        }, follow=True)

        self.assertContains(response, 'deleted successfully', status_code=200)
        self.assertFalse(Voter.objects.filter(id=voter.id).exists())

    def test_superuser_can_create_candidate_from_ui(self):
        admin_client = Client()
        User.objects.create_superuser(username='admin', email='admin@example.com', password='pass12345')
        admin_client.login(username='admin', password='pass12345')

        response = admin_client.post(reverse('superuser_panel'), {
            'action': 'create_candidate',
            'election': self.election.id,
            'name': 'Rahul Verma',
            'party': 'Future Voice',
            'manifesto': 'Safer campus and better labs',
            'display_order': 2,
        }, follow=True)

        self.assertContains(response, 'added successfully', status_code=200)
        self.assertTrue(Candidate.objects.filter(name='Rahul Verma', election=self.election).exists())

    def test_superuser_can_delete_candidate_from_ui(self):
        admin_client = Client()
        User.objects.create_superuser(username='admin', email='admin@example.com', password='pass12345')
        admin_client.login(username='admin', password='pass12345')

        response = admin_client.post(reverse('superuser_panel'), {
            'action': 'delete_candidate',
            'candidate_id': self.candidate.id,
        }, follow=True)

        self.assertContains(response, 'deleted successfully', status_code=200)
        self.assertFalse(Candidate.objects.filter(id=self.candidate.id).exists())
