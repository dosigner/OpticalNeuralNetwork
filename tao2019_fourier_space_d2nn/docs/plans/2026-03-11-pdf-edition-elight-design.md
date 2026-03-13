# PDF Edition eLight-Style Redesign Design

## Goal

`reports/final_report_fd2nn_pdf.md`를 다시 렌더링할 때, 현재의 범용 기술 문서 스타일 대신 eLight 저널에 가까운 준저널 브로슈어형 시각 언어를 적용한다. 핵심은 "AI가 꾸민 문서"처럼 보이지 않게 하면서도, 학술 리포트보다 약간 더 편집된 첫인상과 그림 중심성을 확보하는 것이다.

## Constraints

- 렌더링 파이프라인은 유지한다.
  - 대상 스크립트: `reports/obsidian_to_pdf.py`
  - 엔진: Markdown -> HTML -> WeasyPrint PDF
- 본문 구조와 문서 내용은 유지한다.
- 이미지, 수식, 표, callout이 이미 안정적으로 렌더되는 현 상태를 깨지 않는다.
- 새로운 외부 툴체인 도입 없이 CSS/HTML 수준에서 해결한다.

## Visual Direction

### 1. Tone

- 저널 본문 톤은 차분하고 절제된 serif 중심
- 첫 페이지와 섹션 시작부는 브로슈어처럼 약간 더 강한 위계
- 과포화된 색상, 카드형 UI, 대형 배지, 그라디언트는 배제

### 2. Typography

- 제목: `EB Garamond` 또는 시스템 serif 계열
- 본문: `Noto Serif CJK KR` 중심
- UI성 메타 정보: `Noto Sans CJK KR`
- 코드: `DejaVu Sans Mono` 유지

### 3. Color

- 기본 잉크색: 짙은 남청/먹색
- 보조선: 연한 회청색
- 강조는 1-2개의 낮은 채도 포인트만 사용
- 기존 callout 색상은 전반적으로 탈색하고 더 인쇄물답게 조정

### 4. Layout

- A4 단일 컬럼 유지
- 여백을 조금 더 넓히고 line-height를 정제
- 표지에 journal kicker, title deck, meta block 추가
- figure caption과 table heading을 학술지 스타일로 정제

## Recommended Implementation

### Approach A: CSS-only refresh

- `build_css()`만 수정
- 장점: 가장 안전
- 단점: title page나 body wrapper 구조 제어가 제한됨

### Approach B: CSS + lightweight HTML wrappers

- `build_css()`, `build_title_page()`, `build_html()`를 함께 수정
- 본문을 `journal-shell`, `article-body` 같은 wrapper에 넣고 표지 구조를 분리
- 장점: 스타일 완성도가 높고 리스크도 낮음
- 단점: 테스트가 조금 더 필요

### Approach C: Full editorial template

- section opener, abstract area, running header까지 적극 변경
- 장점: 시각적 임팩트 최대
- 단점: WeasyPrint pagination 리스크가 높고 과장된 느낌이 나기 쉬움

## Decision

Approach B를 채택한다.

이유:
- 사용자가 원하는 "준저널 브로슈어형"을 가장 자연스럽게 구현할 수 있다.
- WeasyPrint에서 안정적으로 유지 가능한 범위다.
- 렌더러 구조를 보존하면서도 title page, wrapper, caption, table, callout의 시각 톤을 충분히 바꿀 수 있다.

## Acceptance Criteria

- 표지가 기존 중앙정렬 기술문서형이 아니라 저널형 title deck으로 보인다.
- 본문은 serif 중심의 읽기감을 가지며, 과한 카드형 UI 느낌이 없다.
- 표와 callout은 인쇄형 학술 문서에 가까운 절제된 형태로 보인다.
- `reports/final_report_fd2nn_pdf.pdf`가 새 스타일로 다시 생성된다.
- 렌더된 주요 페이지를 PNG로 확인했을 때 제목, 표, 그림 캡션, callout이 깨지지 않는다.
