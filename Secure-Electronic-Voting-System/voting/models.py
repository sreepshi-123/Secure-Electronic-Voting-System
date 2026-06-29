from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone


class Voter(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True)
    voter_id = models.CharField(max_length=50, unique=True)
    is_active = models.BooleanField(default=True)
    is_approved = models.BooleanField(default=True)
    registered_at = models.DateTimeField(default=timezone.now, editable=False)
    last_verified_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} - {self.voter_id}"

    @property
    def has_voted(self):
        return self.vote_records.exists()


class FaceBiometricSample(models.Model):
    voter = models.ForeignKey(Voter, on_delete=models.CASCADE, related_name='face_samples')
    embedding = models.TextField()
    quality_score = models.FloatField(default=0.0)
    brightness_score = models.FloatField(default=0.0)
    blur_score = models.FloatField(default=0.0)
    is_reference = models.BooleanField(default=False)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-captured_at']

    def __str__(self):
        return f"Face sample for {self.voter.voter_id} at {self.captured_at:%Y-%m-%d %H:%M}"


class IrisBiometricSample(models.Model):
    voter = models.ForeignKey(Voter, on_delete=models.CASCADE, related_name='iris_samples')
    left_features = models.TextField()
    right_features = models.TextField()
    quality_score = models.FloatField(default=0.0)
    is_reference = models.BooleanField(default=False)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-captured_at']

    def __str__(self):
        return f"Iris sample for {self.voter.voter_id} at {self.captured_at:%Y-%m-%d %H:%M}"


class Election(models.Model):
    STATUS_DRAFT = 'draft'
    STATUS_ACTIVE = 'active'
    STATUS_CLOSED = 'closed'
    STATUS_CHOICES = [
        (STATUS_DRAFT, 'Draft'),
        (STATUS_ACTIVE, 'Active'),
        (STATUS_CLOSED, 'Closed'),
    ]

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    starts_at = models.DateTimeField()
    ends_at = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-starts_at', '-created_at']

    def __str__(self):
        return self.name

    def clean(self):
        if self.ends_at <= self.starts_at:
            raise ValidationError('Election end time must be after the start time.')

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)
        if self.status == self.STATUS_ACTIVE:
            Election.objects.exclude(pk=self.pk).filter(status=self.STATUS_ACTIVE).update(status=self.STATUS_CLOSED)

    @property
    def is_open(self):
        now = timezone.now()
        return self.status == self.STATUS_ACTIVE and self.starts_at <= now <= self.ends_at


class Candidate(models.Model):
    election = models.ForeignKey(Election, on_delete=models.CASCADE, related_name='candidates')
    name = models.CharField(max_length=150)
    party = models.CharField(max_length=150)
    manifesto = models.TextField(blank=True)
    display_order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['display_order', 'name']
        unique_together = [('election', 'name')]

    def __str__(self):
        return f"{self.name} ({self.party})"


class VoteRecord(models.Model):
    voter = models.ForeignKey(Voter, on_delete=models.CASCADE, related_name='vote_records')
    election = models.ForeignKey(Election, on_delete=models.CASCADE, related_name='vote_records')
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='vote_records')
    cast_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-cast_at']
        constraints = [
            models.UniqueConstraint(fields=['voter', 'election'], name='unique_vote_per_voter_per_election'),
        ]

    def __str__(self):
        return f"{self.voter.voter_id} -> {self.candidate.name} ({self.election.name})"


class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('register_attempt', 'Register Attempt'),
        ('register_success', 'Register Success'),
        ('election_create', 'Election Created'),
        ('voter_delete', 'Voter Deleted'),
        ('vote_attempt', 'Vote Attempt'),
        ('vote_success', 'Vote Success'),
        ('vote_rejected', 'Vote Rejected'),
        ('auth_login', 'Auth Login'),
        ('auth_logout', 'Auth Logout'),
    ]

    STATUS_CHOICES = [
        ('success', 'Success'),
        ('failure', 'Failure'),
        ('info', 'Info'),
    ]

    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='info')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    voter = models.ForeignKey(Voter, on_delete=models.SET_NULL, null=True, blank=True)
    election = models.ForeignKey(Election, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.CharField(max_length=45, blank=True)
    user_agent = models.CharField(max_length=255, blank=True)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.action} ({self.status}) at {self.created_at:%Y-%m-%d %H:%M}"
