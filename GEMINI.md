# ARIES — AI-Enhanced SOAR Platform

ARIES is an AI-Enhanced Security Orchestration, Automation, and Response (SOAR)
platform. Its purpose is real-time alert triage, IOC extraction, and incident
summarization for security operations centres (SOCs).

This context file is the **root** for all Gemini CLI sessions run from this
repository. Sub-files are imported below using the `@` syntax and are loaded
on-demand via JIT context loading.

---

## Mandatory reading before any task

Always orient yourself with the full design before making changes:

- `.aiglobal/specs/REQUIREMENTS.md` — 28 functional requirements (REQ-01..28)
- `.aiglobal/specs/DESIGN.md` — System architecture, domain model, data design
- `.aiglobal/specs/TASKS.md` — Master task list (T-001..028)
- `.aiglobal/AI_DESIGN.md` — AI/ML pipeline design specification

---

## Project sub-context files

@.gemini/context/architecture.md
@.gemini/context/ml-pipelines.md
@.gemini/context/fastapi-service.md
@.gemini/context/infrastructure.md
@.gemini/context/wazuh-integration.md
@.gemini/context/coding-conventions.md
