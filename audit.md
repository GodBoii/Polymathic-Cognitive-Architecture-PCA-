# FULL-DEPTH CODEBASE AUDIT —> 

## Your Role

You are performing a forensic, exhaustive audit of this entire codebase — not a skim, not a "top 5 issues" pass. Treat this like a full-body diagnostic scan: every file is a tissue sample, every function is a cell, and nothing gets cleared as "probably fine" without you actually checking it. Surface-level bugs (typos, missing null checks) should be in your report, but so should the things that only show up when you trace execution across files, cross-reference assumptions between modules, or actually run the code and watch what happens.

**Non-negotiable rule:** Every claim in your final report must be backed by an exact file path and line number (or range), and where relevant, a short code excerpt. "This might be an issue" is not acceptable — either show the trace that proves it's an issue, or label it explicitly as a hypothesis that needs verification, with the steps to verify it.

Do not summarize from filenames or function names alone. Open and read the actual implementation before describing what something does.

---

## PHASE 0 — Inventory & Setup

1. Map the full directory tree. Identify the tech stack (languages, frameworks, package managers, build tools, runtime versions).
2. Read all config files: `package.json`/`requirements.txt`/`pom.xml`/etc., `.env.example`, CI/CD configs, linter/formatter configs, Docker files, infra-as-code.
3. Identify entry points (main files, server bootstrap, app root) and trace what loads first, second, third.
4. List every external dependency and service integration (databases, third-party APIs, queues, cache layers, auth providers, payment processors, etc.) and note their versions and whether they are current/deprecated/vulnerable.
5. Try to actually install dependencies and run the project (dev server, build, tests). Record every warning and error verbatim, even ones you're tempted to dismiss as "just noise."

---

## PHASE 1 — Architecture & System Design

1. Draw out (in text/diagram form) the overall architecture: layers, services, modules, and how they communicate (REST, GraphQL, RPC, events, websockets, direct imports).
2. Identify the design pattern(s) in use (MVC, layered, hexagonal, microservices, monolith, etc.) and **whether the code actually follows the pattern it claims to**, or has drifted (e.g., business logic leaking into controllers/components).
3. Map data flow end-to-end for at least 3 core user-facing features: input → validation → processing → storage → response → UI update.
4. Identify circular dependencies, tight coupling between modules that should be independent, and any "god files/classes" doing too much.
5. Note inconsistencies in architecture across the codebase (e.g., half the routes use one error-handling style, the other half use another).

---

## PHASE 2 — Backend Deep Dive

For every backend module/service/route:

1. List every function and class. For each: what it does, what it assumes about its inputs, what it returns, what can make it throw or fail silently.
2. Trace every API endpoint: method, route, middleware chain, auth requirements, input validation, what happens on malformed/missing/malicious input, what status codes and payloads are returned in success and every failure branch.
3. Check transaction handling and data consistency — what happens if a multi-step operation fails halfway through (partial writes, orphaned records, no rollback).
4. Check concurrency handling — race conditions, missing locks, shared mutable state, idempotency of operations that should be idempotent (e.g., payment, retries).
5. Check all database queries: N+1 query problems, missing indexes implied by query patterns, unparameterized queries, overly broad queries (`SELECT *`), missing pagination on large datasets.
6. Check logging: are errors actually logged with enough context to debug in production, or silently swallowed? Search specifically for empty catch blocks and generic `catch (e) {}` patterns.
7. Check configuration/env var handling — what happens if a required env var is missing? Does it fail loudly at startup or fail mysteriously later?

---

## PHASE 3 — Frontend / UI-UX Deep Dive

1. Inventory every screen/page/component. For each, list every interactive element (buttons, forms, modals, dropdowns, toggles, drag-and-drop, etc.) and what each one is supposed to do vs. what it actually does in code.
2. Check every button and action for: disabled states during loading, double-submit protection, what happens on slow network / failed request / partial response.
3. Check all forms: validation rules (client AND server side — do they match?), error message clarity, what happens with empty/extreme/special-character/emoji/very long input, autofill behavior, accessibility of error states.
4. Check all animations/transitions: are they interrupted gracefully if the user navigates away mid-animation? Do they cause layout shift, janky reflows, or block interaction while playing?
5. Check responsive behavior across breakpoints (mobile/tablet/desktop) and orientation changes.
6. Check loading states, empty states, and error states for every data-driven view — does every screen have all three, or do some just show a blank screen?
7. Check accessibility: semantic HTML, ARIA attributes, keyboard navigation, focus management (especially in modals/drawers), color contrast, screen-reader behavior.
8. Check state management: where is state held, what triggers re-renders, are there stale-state bugs (UI showing data that no longer matches the source of truth), memory leaks from uncleaned subscriptions/listeners/timers.
9. Check browser/back-button behavior, deep-linking, and what happens on page refresh mid-flow.

---

## PHASE 4 — Security, Privacy & Auth

1. Trace the full authentication flow: signup, login, logout, session/token refresh, password reset, account recovery. Identify every place a token/session is created, validated, or invalidated.
2. Check authorization on every endpoint and UI action — not just "is the user logged in" but "is this specific user allowed to do this specific thing to this specific resource" (look hard for IDOR-style issues: can user A access/modify user B's data by changing an ID in a request?).
3. Check input sanitization against injection classes relevant to the stack (SQL/NoSQL injection, XSS, command injection, path traversal, SSRF) — identify where user input reaches a query, a shell command, a file path, or is rendered unescaped.
4. Check secrets handling: are any API keys, credentials, or tokens hardcoded or logged? Are secrets exposed to the frontend that shouldn't be?
5. Check CORS, CSRF protections, rate limiting, and brute-force protection on sensitive endpoints (login, password reset, OTP).
6. Check what personal/sensitive data is collected, where it's stored, whether it's encrypted at rest/in transit, and whether it's exposed in API responses, logs, or error messages more broadly than necessary.
7. Check dependency vulnerabilities (known CVEs in current versions of libraries used).
8. Check file upload handling (if any): type validation, size limits, storage location, executable content risk.

---

## PHASE 5 — Correlations, Integrations & Cross-Cutting Concerns

1. Identify every place where frontend assumptions about backend behavior (data shape, error format, status codes) could silently diverge from what the backend actually does — these mismatches are often invisible until a specific edge case hits production.
2. Identify shared utilities/constants that are duplicated instead of shared (copy-pasted logic that has already or will eventually drift out of sync).
3. Identify implicit contracts between modules that aren't enforced by types/schemas/tests (e.g., one module assumes a field is always present, another module that produces that data doesn't guarantee it).
4. Check third-party integration failure handling — what happens when an external API is slow, down, or returns unexpected data?
5. Check feature flags / environment-specific logic for cases where dev/staging/prod behavior could silently diverge.

---

## PHASE 6 — Edge Cases & Error Handling (the "easily missed" layer)

This is the phase most audits skip. Specifically hunt for:

1. Off-by-one errors, boundary conditions (0, negative numbers, empty arrays/strings, exactly-at-limit values).
2. Timezone, locale, and date/time edge cases (DST transitions, leap years, different date formats).
3. Unicode/emoji/multi-byte character handling in any text field, especially around length limits and storage.
4. What happens with zero items, exactly one item, and the maximum expected number of items in any list/pagination.
5. Network edge cases: slow connections, dropped connections mid-request, duplicate requests due to retries, out-of-order responses.
6. Concurrent users acting on the same resource at the same time.
7. Clock skew and stale cache assumptions.
8. Silent failures — anywhere an error is caught but the user/system isn't informed and the app continues as if nothing happened.
9. Resource exhaustion — what happens with a very large input file, a very long list, a deeply nested object, a very long-running request.
10. Any TODO, FIXME, HACK, or commented-out code — read the surrounding context and assess whether it indicates a real unresolved problem.

---

## PHASE 7 — Actually Test It (don't just read it)

1. Run any existing automated tests. Report pass/fail and, critically, **assess test quality** — are they testing real behavior or just asserting trivial things? What's NOT covered?
2. If feasible, manually exercise the core user flows end-to-end and document actual observed behavior, including anything that doesn't match what the code/comments/docs claim it should do.
3. Try deliberately malformed, boundary, and adversarial inputs at every entry point you can reach (forms, API calls, query params) and record actual results.
4. If there's no test suite or it's thin, explicitly call that out as a finding, not just a gap.

---

## PHASE 8 — Code Quality & Maintainability

1. Identify dead code, unreachable branches, and unused exports/imports/variables.
2. Identify inconsistent naming, inconsistent error-handling style, inconsistent formatting that linting should have caught but didn't.
3. Identify functions/files that are too large or doing too many unrelated things.
4. Identify missing or misleading comments/docstrings (especially comments that describe behavior the code no longer actually has — these are landmines for the next person).
5. Identify hardcoded "magic" values that should be named constants/config.

---

## PHASE 9 — Performance

1. Identify obvious performance issues: unnecessary re-renders, unbounded loops over large data, blocking operations on the main/UI thread, unoptimized assets (oversized images, unminified bundles), missing memoization/caching where it would clearly help.
2. Identify any operation whose cost scales poorly with data size (e.g., O(n²) where it matters) in a hot path.

---

## FINAL DELIVERABLE — Report Format

Produce a single structured report with these sections:

### 1. Executive Summary
- One paragraph: overall health of the codebase.
- A table of all findings grouped by severity: **Critical / High / Medium / Low / Cosmetic**, with a count per category.

### 2. Architecture Overview
- Plain-language description of the system as it actually is (not as documented, if the two differ).
- Diagram or structured outline of components and data flow.

### 3. Detailed Findings
For **every** issue found, regardless of severity, use this format:

```
ID: [sequential number]
Title: [short description]
Severity: Critical / High / Medium / Low / Cosmetic
Category: [Security / Logic / UI-UX / Performance / Architecture / Data Integrity / Maintainability / Testing / Other]
Location: file path + line number(s)
Description: what is wrong, in plain language
Evidence: relevant code excerpt and/or reproduction steps / observed test result
Impact: what breaks, for whom, under what conditions
Recommended Fix: specific, actionable
Confidence: High / Medium / Low (and why, if not High)
```

### 4. "Easily Missed" Findings — Called Out Separately
A dedicated subsection specifically for subtle issues: race conditions, edge cases, silent failures, cross-module contract mismatches, things that only surface under specific timing/data/concurrency conditions. Don't bury these in the general list — they're the whole point of this audit.

### 5. Security & Privacy Summary
Separate rollup of every security/privacy/auth finding regardless of where it appeared above, since these need focused attention.

### 6. Test Coverage Assessment
What's tested, what's not, what the highest-value next tests would be.

### 7. Prioritized Action Plan
An ordered list of what to fix first and why, considering both severity and effort.

---

## Ground Rules

- Do not pad the report with generic best-practice advice that doesn't apply to this specific code. Every finding must be specific to what you actually found here.
- Do not skip files because they "look standard" — boilerplate is exactly where copy-pasted bugs hide.
- If something is ambiguous or you can't fully verify it without more context (e.g., production environment specifics), say so explicitly rather than guessing silently.
- If the codebase is large, work through it systematically (module by module) rather than sampling — note explicitly if you had to skip or skim anything due to scope, and what was skipped.
- Be honest if something is actually fine — the goal is an accurate diagnosis, not maximizing the issue count.