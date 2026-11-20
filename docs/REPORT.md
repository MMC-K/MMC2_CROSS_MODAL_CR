# 상호참조 해결 모델 성능 분석서

이 저장소의 학습·평가 조건과 결과를 설명한 조병길 연구원(KETI)의 기술문서를 첨부한다. 실제 대화 21개 발화, 입력 prompt·기대 출력, JSON 저장 구조, 네 평가 지표의 계산, 성공·오류 사례를 포함한다.

- [성능 분석서 PDF](reports/coreference_performance_analysis_v1.pdf)
- [성능 분석서 Word](reports/coreference_performance_analysis_v1.docx)

문서 제목은 「멀티모달 대화 기반 상호참조 해결 모델 성능 분석서」이며, 형광펜 표시를 제거한 최종본이다. 두 첨부파일은 이 프로젝트에서 작성한 원본의 내용과 동일하다.

분석서 부록 A는 별도로 공유했던 ZIP의 압축 해제부터 설명한다. GitHub에서 사용할 때는 그 압축 해제 단계를 이 저장소의 `git clone`으로 대체하고, 같은 데이터와 scripts/ 명령을 사용한다. 실행 절차는 현재 README를 기준으로 따른다.
