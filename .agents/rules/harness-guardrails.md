---
trigger: glob
globs: "**/.agents/**, **/AGENTS.md, **/GEMINI.md, **/SKILL.md, **/hooks.json"
---
# 하네스 파일 작성 가드레일

1. `SKILL.md`는 `name`(상위 폴더명과 일치하는 소문자 `kebab-case`)과 `description` 프론트매터가 필수이며, 본문은 500줄 이하로 유지합니다.
2. 디렉터리 배치 `AGENTS.md`/`GEMINI.md`는 YAML 프론트매터를 절대 사용하지 않으며, 파일당 24,000 bytes 이하(권장 60~100줄)로 유지합니다.
3. `.agents/rules/*.md`의 `globs`에는 `{a,b}` 브레이스 확장을 사용하지 않고 쉼표로 구분된 단일 문자열로 펼쳐 작성합니다.
4. `.agents/agents/*.md`는 `subagent: true`, `mainAgent: false`, 명시적 `tools` 배열이 필수이며 내장 에이전트명(`self`, `browser`, `research`, `image-generator`)과 충돌하지 않아야 합니다.
5. `.agents/hooks.json`의 `PreToolUse`/`PostToolUse`는 `{ "matcher": "...", "hooks": [...] }` 그룹 구조를 반드시 준수합니다.
