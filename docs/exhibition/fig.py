import matplotlib; matplotlib.use('Agg')
import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
W,H=16.5,4.1
fig=plt.figure(figsize=(W/2.54,H/2.54),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
C1,C2,C3='#dbe8f7','#e3f1e0','#fbe7d3'
def box(x,y,w,h,t,c,b=False,fs=6.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=c,ec='#333',lw=0.7))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if b else 'normal',linespacing=1.25)
def arr(x1,y1,x2,y2):
    ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='#333',mutation_scale=7))
yT,yB,h=2.55,0.55,1.05
box(0.15,yT,1.75,h,'뉴스 본문\n(발행 시각)',C1)
box(2.25,yT,2.2,h,'GPT-4o-mini\n요약·감성·이벤트 추출',C1)
box(4.8,yT,2.3,h,'임베딩(1536)+감성\n→ 주 신호 512',C1,True)
box(0.15,yB,1.75,h,'기술적 지표\n9종 × 30일',C2)
box(2.25,yB,2.2,h,'TCN\n(미래 정보 차단)',C2)
box(4.8,yB,2.3,h,'보조 신호\n30 × 512',C2,True)
for y in (yT,yB):
    arr(1.9,y+h/2,2.25,y+h/2); arr(4.45,y+h/2,4.8,y+h/2)
box(7.6,1.35,2.25,1.45,'비대칭\n교차 어텐션\nQ=뉴스, K·V=지표',C3,True)
arr(7.1,yT+h/2,7.6,2.45); arr(7.1,yB+h/2,7.6,1.7)
box(10.25,1.5,1.9,1.15,'게이트 기반\n잔차 결합',C3,True)
arr(9.85,2.075,10.25,2.075)
ax.annotate('',(11.2,2.65),(7.1,yT+h*0.85),arrowprops=dict(arrowstyle='-|>',lw=0.6,color='#777',ls='--',mutation_scale=6,connectionstyle='angle,angleA=0,angleB=90,rad=0'))
box(12.55,1.5,1.6,1.15,'MLP\n(3층)',C3,True)
arr(12.15,2.075,12.55,2.075)
box(14.55,1.5,1.8,1.15,'초과수익률\n(등락) 예측',C3,True)
arr(14.15,2.075,14.55,2.075)
ax.text(12.0,0.8,'뉴스가 질의(Q)로 과거 시장 상태를 선택 참조',ha='center',va='center',fontsize=5.8,color='#555')
fig.savefig('photo.png',dpi=300); print('ok')
