# Pricing Plan Implementation Tracker

This document tracks the implementation of Basic, Intermediate, and Enterprise pricing tiers for the Sinhala Educational Assistant.

## Current Status

- Status: Core pricing, usage limits, admin tier updates, database-backed plan limits, and tests are complete
- Owner: TBD
- Started: 2026-04-30
- Main user field: `users.tier`
- Proposed permission field: `users.role`
- Current default tier in code: `basic`
- Target default tier: `basic`
- Verification: `.\venv\Scripts\python.exe -m pytest tests\test_evaluation_limits.py tests\test_pricing_admin_api.py` passes.

## Target Plans

| Tier | Price | Badge | Learning Limit | Evaluation Session Limit | Per-Session Evaluation Limit |
| --- | --- | --- | --- | --- | --- |
| `basic` | Free forever | Starter | 5 requests/hour | 1 session/day | 10 evaluations/session |
| `intermediate` | 5000 LKR/tier | Most Popular | 20 requests/hour | 5 sessions/day | TBD |
| `enterprise` | 10000 LKR onwards/tier | Best for Scale | 50 requests/hour | 10 sessions/day | Extra evaluations billable |

## Decisions

- [x] Evaluation daily limits reset by calendar day in `Asia/Colombo`.
- [x] Intermediate currently has only 5 evaluation sessions/day and no per-session evaluation count limit.
- [x] Enterprise currently has 10 evaluation sessions/day and allows per-session evaluation overage.
- [x] Tier changes are currently admin-managed through the user tier update endpoint.
- [x] Old tiers are mapped as `normal -> basic`, `classroom -> intermediate`, `institution -> enterprise`.
- [x] Should the first admin user be created by migration, seed script, or manual database update?
- [x] Admins should be able to edit plan limits from the database.

## Future Decisions

- [ ] Should users be able to upgrade tier directly through a payment flow?
- [ ] Should Enterprise overage trigger billing automatically or require later admin review?

## Tier Vs Role

Keep subscription tier and authorization role separate.

| Field | Purpose | Example Values |
| --- | --- | --- |
| `users.tier` | Controls usage limits and pricing plan | `basic`, `intermediate`, `enterprise` |
| `users.role` | Controls permissions and admin access | `user`, `admin` |

An Enterprise user should not automatically become an admin. An admin can manage users, tiers, and future pricing settings without needing an Enterprise subscription.

## Implementation Checklist

### Phase 1: Plan Configuration

- [x] Create a single source of truth for pricing plans, for example `app/core/pricing_plans.py`.
- [x] Define internal tier keys: `basic`, `intermediate`, `enterprise`.
- [x] Include marketing metadata needed by frontend: name, description, badge, price, CTA, note, features.
- [x] Include enforceable limits: learning requests/hour, evaluation sessions/day, evaluations/session, overage behavior.

### Phase 2: User Tier Model And Migration

- [x] Update `app/shared/models/user.py` so `tier` defaults to `basic`.
- [x] Create Alembic migration to change database default to `basic`.
- [x] Migrate existing rows:
  - [x] `normal` -> `basic`
  - [x] `classroom` -> `intermediate`
  - [x] `institution` -> `enterprise`
- [x] Expose `tier` in `app/schemas/user.py` response models if frontend needs it.

### Phase 3: Admin Role And Permissions

- [x] Add `role` column to `app/shared/models/user.py`.
- [x] Use default role `user`.
- [x] Create Alembic migration for `users.role`.
- [x] Decide how the first admin account is created.
- [x] Expose `role` in user response only if needed by frontend.
- [x] Add helper/dependency such as `require_admin_user`.
- [x] Protect tier-management endpoints with admin-only access.
- [ ] Protect future plan-management endpoints with admin-only access when those endpoints are added.

### Phase 4: Usage Service

- [x] Refactor `app/services/usage_service.py` to use the new plan config.
- [x] Add `check_learning_request_limit(user_id)`.
- [x] Add `check_evaluation_session_limit(user_id)`.
- [x] Add `check_evaluations_per_session_limit(user_id, answer_resource_ids)`.
- [x] Return clear HTTP 403 errors with current tier and limit details.
- [x] Decide whether unknown tier should fallback to `basic` or fail loudly.

### Phase 5: Learning Mode Enforcement

- [x] Add learning limit check to `app/routers/messages.py` before user message creation.
- [x] Add learning limit check to voice learning endpoints in `app/components/voice_qa/routers/voice_router.py`.
- [x] Make sure only user-created learning requests count, not assistant/system messages.
- [x] Make sure evaluation-mode chat messages do not consume learning quota unless intentionally required.

### Phase 6: Evaluation Mode Enforcement

- [x] Replace current evaluation limit check in `app/services/evaluation/evaluation_workflow_service.py`.
- [x] Enforce daily evaluation session limit before creating or starting a new evaluation session.
- [x] Enforce Basic per-session evaluation limit of 10 answer scripts.
- [x] Implement Enterprise overage behavior.
- [x] Avoid double-counting reused evaluation sessions if `process_documents` created a session earlier.

### Phase 7: API Endpoints

- [x] Add `GET /api/v1/pricing/plans`.
- [x] Add `GET /api/v1/usage/me`.
- [x] Add admin-only tier update flow.
- [ ] Optionally add payment-webhook tier update flow later.
- [x] Register new router in `app/api/v1/router.py`.

### Phase 8: Admin-Managed Plan Limits

- [x] Add database table for editable pricing plans.
- [x] Add admin-only endpoints for listing and updating pricing plans.
- [x] Track who updated plan limits with `updated_by`.
- [x] Track when plan limits changed with `updated_at`.
- [ ] Keep audit history if pricing changes affect billing or contractual limits.

Phase 8 is implemented for database-backed plan edits. `app/core/pricing_plans.py` remains the fallback/default catalog used to seed missing rows and keep tests resilient before migrations run.

Possible future table:

| Column | Purpose |
| --- | --- |
| `tier_key` | Stable tier identifier |
| `name` | Display name |
| `price_lkr` | Fixed LKR price, if applicable |
| `price_lkr_from` | Starting price for Enterprise |
| `learning_requests_per_hour` | Learning quota |
| `evaluation_sessions_per_day` | Daily session quota |
| `evaluations_per_session` | Per-session answer/evaluation quota |
| `allow_overage` | Whether extra usage can continue |
| `is_active` | Whether the plan is available |
| `updated_by` | Admin user who last changed the plan |
| `updated_at` | Last update timestamp |

### Phase 9: Tests

- [x] Test Basic learning limit: 5 requests/hour allowed, 6th blocked.
- [x] Test Intermediate learning limit: 20 requests/hour.
- [x] Test Enterprise learning limit: 50 requests/hour.
- [x] Test Basic evaluation sessions: 1/day allowed, 2nd blocked.
- [x] Test Intermediate evaluation sessions: 5/day.
- [x] Test Enterprise evaluation sessions: 10/day.
- [x] Test Basic evaluations/session: 10 allowed, 11th blocked.
- [x] Test migration maps old tiers correctly.
- [x] Test pricing plans endpoint returns all required frontend metadata.
- [x] Test normal users cannot update tiers.
- [x] Test admins can update user tiers.
- [x] Test admin-only endpoints reject unauthenticated users.
- [x] Test admin-only endpoints reject authenticated non-admin users.
- [x] Test unauthenticated users cannot list admin pricing plans.
- [x] Test admins can update pricing plan limits.

## Code Touchpoints

- `app/shared/models/user.py`
- `app/schemas/user.py`
- `app/core/security.py`
- `app/core/pricing_plans.py`
- `app/shared/models/pricing_plan.py`
- `app/services/pricing_plan_service.py`
- `app/services/usage_service.py`
- `app/routers/messages.py`
- `app/components/voice_qa/routers/voice_router.py`
- `app/services/evaluation/evaluation_workflow_service.py`
- `app/api/v1/router.py`
- `app/core/config.py`
- `migrations/versions/`
- `tests/`

## Notes

- Learning requests reset every hour.
- Evaluation limits reset daily by calendar day in `Asia/Colombo`.
- Enterprise customers can continue past the per-session evaluation count limit; billing/review for that overage is future work.
- `UsageService` reads pricing limits through `PricingPlanService`, so database edits affect quota enforcement.
- Admin-editable plan limits are available through `GET /api/v1/pricing/admin/plans` and `PATCH /api/v1/pricing/admin/plans/{tier}`.
- Test coverage currently includes learning limits, evaluation session limits, Basic per-session evaluation limits, tier migration mapping, pricing metadata, admin-only tier update behavior, and admin pricing-plan updates.
