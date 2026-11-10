"""Write a Korean summary and figures from the completed aggregate comparison."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd

FIELDS = [('physics','물리'),('chemistry','화학'),('biology','생물'),('mathematics','수학')]
METHODS = ('classification','shared_references','citation_network_external','citation_network_local')


def fmt(x):
    return f'{x:+.3f}'.replace('-', '−')


def ci(row, key):
    return f"{fmt(row[key])} [{fmt(row.lo)}, {fmt(row.hi)}]"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','pilot','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    for f in args.analysis.iterdir():
        if f.is_file() and f.suffix in ('.csv','.json'):
            shutil.copyfile(f,args.out/f.name)
    a=args.analysis
    d=json.loads((a/'diagnostics.json').read_text())
    assert d['mode']=='full' and d['windows']==19849 and d['publications']==117602 and d['faculty']==2985
    assert d['original_frozen_sha256']==d['matched_sample_sha256']
    reg=pd.read_csv(a/'slopes.csv').set_index(['subject','method'])
    delta=pd.read_csv(a/'slope_differences.csv').query("versus=='citation_network_local'").set_index('subject')
    fixed=pd.read_csv(a/'fixed_pivots_slope_differences.csv').query("versus=='citation_network_local'").set_index('subject')
    corr=pd.read_csv(a/'correlations.csv').query("left=='citation_network_external' and right=='citation_network_local'")
    def differentiating(data):
        return ', '.join(label for field,label in FIELDS if data.loc[field].lo * data.loc[field].hi > 0) or '없음'
    L=['# 외부 피인용을 추가한 공통 표본 비교','',
       '2026-11-08 · 인용 논문 출판 마감 2025-12-31','',
       f"그래프를 구성하는 핵심 논문 {d['graph_focal_works']:,}편의 OpenAlex 3년 창 조회를 완료했다. "
       f"기존 표본과 동일한 교원 {d['faculty']:,}명, 고유 논문 {d['publications']:,}편, 교원별 2년 창 {d['windows']:,}개를 세 방법으로 다시 계산했다. 논문 목록의 해시가 기존 표본과 일치한다.",'',
       f"OpenAlex 응답에서 방향 간선 {d['acquisition']['edges']:,}개를 확인했다. "
       f"기존 로컬 캐시에 없던 인용 논문 {d['external_citing_works']:,}편의 간선 {d['external_links']:,}개를 추가했다. "
       f"참고문헌 목록에서 대상 논문을 확인하지 못한 API 응답 {d['acquisition']['unresolved_result_records']:,}건은 간선에서 제외했다. "
       f"단일 연결 이웃을 가지치기한 뒤 최단경로 그래프에는 간선 {d['edges_increase_after_pruning']:,}개가 순증했다.",'',
       '표는 임용 후 0–29년의 개인 내 기울기다. 단위는 10년당 표준편차 변화이며 대괄호는 같은 교원을 묶어 재표집한 95% 구간이다. 마지막 열은 동일한 표본에서 외부 피인용을 추가하기 전후의 그래프 기울기 차이다.','',
       '| 분야 | OpenAlex 분류 | 공유 참고문헌 | 외부 피인용 추가 그래프 | 추가 후 − 로컬 그래프 |',
       '| --- | ---: | ---: | ---: | ---: |']
    for field,label in FIELDS:
        L.append('| '+' | '.join([label,*[ci(reg.loc[(field,m)],'sd_per_decade') for m in METHODS[:3]],
                                  ci(delta.loc[field],'sd_per_decade_delta')])+' |')
    L += ['',f"외부 연결 추가 전후의 구간 퍼짐 순위 상관은 분야별 {corr.spearman.min():.3f}–{corr.spearman.max():.3f}였다. "
          f"기울기 차이의 95% 구간이 0을 포함하지 않는 분야는 피벗을 새로 고르면 {differentiating(delta)}, 기존 피벗을 쓰면 {differentiating(fixed)}였다.",'',
          '## 설계와 범위','',
          '- 피인용은 대상 논문의 출판 다음 해부터 세 번째 해까지의 완결 연도에 관측된 실제 방향 간선이다. 마지막 인용 논문 출판연도는 2025년이다. OpenAlex API는 2026-11-08에 조회했다.',
          '- 세 방법은 교원·구간별로 동일한 논문을 사용했다. 공유 참고문헌에는 기존 TF–IDF/SVD 좌표를, 그래프에는 150개 피벗과 64차원 MDS를 사용했다. 그래프에 추가한 외부 피인용은 기존 로컬 캐시 밖의 인용 논문에서 온 연결이다.',
          '- 구간별 곡선은 기술 평균이다. 기울기는 임용 후 0–29년 안에서 둘 이상의 관측 구간이 있는 교원의 개인 내 추세이며, 임용의 인과효과를 추정하지 않는다. 그래프의 피인용 관측 범위는 OpenAlex 색인에 의존한다.',
          '', '[경력 구간별 곡선과 기울기 비교(PDF)](external_openalex_full_comparison.pdf)']
    (args.out/'README.md').write_text('\n'.join(L)+'\n')
    s=(args.pilot/'plot_results.py').read_text()
    s=s.replace('Topic spread in the OpenAlex pilot','Topic spread after adding OpenAlex citations')
    s=s.replace('g.n >= 10','g.n >= 30').replace('[-6,0,6,12,18]','[-6,0,6,12,18,24,30,36]').replace('(-6.6,19.5)','(-6.6,36.5)')
    s=s.replace('Means shown for windows with 10 or more researchers. The pilot is a subset of the original study.',
                'Means shown for windows with 30 or more researchers. The researchers observed can differ across windows.')
    s=s.replace('Years 0–19 after first appointment','Years 0–29 after first appointment')
    s=s.replace('External links were collected for pilot papers and added to the original local graph.',
                'The four series use the same papers. External OpenAlex links are added to the original local graph.')
    s=s.replace('external_openalex_comparison.pdf','external_openalex_full_comparison.pdf')
    (args.out/'plot_results.py').write_text(s)
    print('\n'.join(L[9:18]))


if __name__=='__main__':
    main()
