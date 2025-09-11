---
name: Quality Gate Failure
about: Report a quality gate failure in CI/CD
title: '[QUALITY] Quality gate failure in workflow'
labels: 'quality, ci/cd'
assignees: ''

---

## Quality Gate Failure Report

**Workflow:** [e.g., Code Quality & Coverage]
**Job:** [e.g., code-quality, coverage]
**Branch:** [e.g., main, feature/xyz]

### Failure Details
<!-- Describe what quality gate failed -->

### Error Messages
<!-- Paste relevant error messages or logs -->

```
[Paste error messages here]
```

### Steps to Reproduce
1. 
2. 
3. 

### Expected Behavior
<!-- What should have happened -->

### Additional Context
<!-- Add any other context about the problem here -->

### Checklist
- [ ] Code formatting (Black)
- [ ] Import sorting (isort)
- [ ] Linting (flake8)
- [ ] Type checking (mypy)
- [ ] Security scanning (bandit)
- [ ] Coverage threshold (80%)
- [ ] Dependency vulnerabilities (safety)