from django.contrib import admin
from .models import (
    AuditLog,
    Candidate,
    Election,
    FaceBiometricSample,
    IrisBiometricSample,
    VoteRecord,
    Voter,
)


class FaceBiometricSampleInline(admin.TabularInline):
    model = FaceBiometricSample
    extra = 0
    readonly_fields = ('quality_score', 'brightness_score', 'blur_score', 'captured_at')


class IrisBiometricSampleInline(admin.TabularInline):
    model = IrisBiometricSample
    extra = 0
    readonly_fields = ('quality_score', 'captured_at')


@admin.register(Voter)
class VoterAdmin(admin.ModelAdmin):
    list_display = ('name', 'voter_id', 'email', 'is_active', 'is_approved', 'has_voted', 'registered_at')
    search_fields = ('name', 'voter_id', 'email')
    list_filter = ('is_active', 'is_approved', 'registered_at')
    inlines = [FaceBiometricSampleInline, IrisBiometricSampleInline]


class CandidateInline(admin.TabularInline):
    model = Candidate
    extra = 1


@admin.register(Election)
class ElectionAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'starts_at', 'ends_at', 'created_at')
    list_filter = ('status',)
    search_fields = ('name',)
    inlines = [CandidateInline]


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ('name', 'party', 'election', 'display_order')
    list_filter = ('election', 'party')
    search_fields = ('name', 'party', 'election__name')
    ordering = ('election', 'display_order', 'name')


@admin.register(VoteRecord)
class VoteRecordAdmin(admin.ModelAdmin):
    list_display = ('voter', 'candidate', 'election', 'cast_at')
    list_filter = ('election', 'candidate')
    search_fields = ('voter__name', 'voter__voter_id', 'candidate__name', 'election__name')


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('action', 'status', 'user', 'voter', 'election', 'created_at')
    list_filter = ('action', 'status', 'election')
    search_fields = ('user__username', 'voter__name', 'voter__voter_id', 'ip_address')
    readonly_fields = ('created_at',)
