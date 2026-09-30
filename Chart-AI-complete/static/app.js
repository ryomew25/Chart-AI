const $ = (id) => document.getElementById(id);
const charts = {};
let lastData = null;
let activeRegion = 'jp';
let screenPage = 0;
let searchTimer = null;

Chart.defaults.font.family = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
Chart.defaults.color = '#91a0b8';

function esc(v){ return String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function num(v, digits=2){ return v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('ja-JP',{maximumFractionDigits:digits}); }
function pct(v){ return v == null || !Number.isFinite(Number(v)) ? '—' : `${Number(v).toFixed(2)}%`; }
function money(v){
  if(v == null || !Number.isFinite(Number(v))) return '—';
  const n=Number(v); const a=Math.abs(n);
  if(a>=1e12) return `${(n/1e12).toFixed(2)}T`;
  if(a>=1e9) return `${(n/1e9).toFixed(2)}B`;
  if(a>=1e6) return `${(n/1e6).toFixed(2)}M`;
  return n.toLocaleString('ja-JP',{maximumFractionDigits:0});
}
function destroy(name){ if(charts[name]){ charts[name].destroy(); delete charts[name]; } }
function line(id, labels, datasets, extra={}){
  destroy(id); const c=$(id); if(!c) return;
  charts[id]=new Chart(c,{type:'line',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:'index'},plugins:{legend:{position:'top',align:'start',labels:{usePointStyle:true,boxWidth:7,padding:14}}},scales:{x:{grid:{display:false},ticks:{maxTicksLimit:9,maxRotation:0}},y:{grid:{color:'rgba(255,255,255,.055)'},border:{display:false}}},elements:{line:{tension:.22,borderWidth:2},point:{radius:0,hoverRadius:4}},...extra}});
}
const ds=(label,data,border,dash=[])=>({label,data,borderColor:border,backgroundColor:'transparent',borderDash:dash,fill:false});

function showView(name){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(v=>v.classList.remove('active'));
  $(`${name}View`).classList.add('active');
  document.querySelector(`.nav-tab[data-view="${name}"]`).classList.add('active');
  if(name==='watchlist') renderWatchlist();
}
document.querySelectorAll('.nav-tab').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.view)));

async function searchSymbols(q){
  const box=$('suggestions');
  if(!q.trim()){box.classList.remove('show');box.innerHTML='';return;}
  try{
    const r=await fetch(`/api/search?q=${encodeURIComponent(q)}`); const data=await r.json();
    if(!Array.isArray(data)||!data.length){box.innerHTML='<div class="suggestion muted">候補が見つかりません</div>';box.classList.add('show');return;}
    box.innerHTML=data.map(x=>`<button class="suggestion" data-symbol="${esc(x.symbol)}"><span><b>${esc(x.name)}</b><small>${esc(x.symbol)} · ${esc(x.exchange)}</small></span><i>↗</i></button>`).join('');
    box.querySelectorAll('.suggestion').forEach(b=>b.addEventListener('click',()=>{ $('ticker').value=b.dataset.symbol;box.classList.remove('show');analyze(); }));
    box.classList.add('show');
  }catch(e){ box.classList.remove('show'); }
}
$('ticker').addEventListener('input',e=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>searchSymbols(e.target.value),260);});
$('ticker').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
document.addEventListener('click',e=>{if(!e.target.closest('.search-wrap')) $('suggestions').classList.remove('show');});
$('analyzeBtn').addEventListener('click',analyze);

function renderMetric(data){
  const c=data.company||{}; const t=data.technical||{}; const f=data.fundamental||{};
  $('emptyState').classList.add('hidden'); $('dashboard').classList.remove('hidden');
  $('companyName').textContent=c.name||data.ticker;
  $('companyMeta').textContent=[data.ticker,c.sector,c.industry].filter(Boolean).join(' · ');
  $('price').textContent=num(data.price);
  $('change').textContent=`${data.change>=0?'+':''}${num(data.change)} (${data.change_percent>=0?'+':''}${num(data.change_percent)}%)`;
  $('change').className=data.change>=0?'positive':'negative';
  $('direction').textContent=t.direction||'—'; $('direction').className=t.direction==='上昇寄り'?'positive':t.direction==='下降寄り'?'negative':'';
  $('upProb').textContent=`${num(t.up_probability,1)}%`; $('downProb').textContent=`下降寄り ${num(t.down_probability,1)}%`;
  $('fundScore').textContent=f.overall_score!=null?`${num(f.overall_score,1)}/100`:'—';
  $('chartMeta').textContent=data.meta?.intraday?'5分足 · 1日':'日足 · '+data.meta?.period;
  $('upReasons').innerHTML=(t.up_reasons||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li>目立った上昇材料なし</li>';
  $('downReasons').innerHTML=(t.down_reasons||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li>目立った下降材料なし</li>';
  renderFundamental(f);
  renderCharts(data.chart||[]);
  updateWatchButton(data.ticker);
}
function renderFundamental(f){
  if(!f||!f.available) return;
  $('growthScore').textContent=num(f.scores.growth,1); $('profitScore').textContent=num(f.scores.profitability,1); $('valuationScore').textContent=num(f.scores.valuation,1); $('healthScore').textContent=num(f.scores.financial_health,1); $('returnScore').textContent=num(f.scores.shareholder_return,1);
  const g=f.growth||{}, p=f.profitability||{}, v=f.valuation||{}, h=f.financial_health||{}, r=f.shareholder_return||{};
  $('revenueGrowth').textContent=pct(g.revenue_growth);$('earningsGrowth').textContent=pct(g.earnings_growth);$('quarterGrowth').textContent=pct(g.earnings_quarterly_growth);
  $('eps').textContent=num(p.eps);$('roe').textContent=pct(p.roe);$('roa').textContent=pct(p.roa);$('opMargin').textContent=pct(p.operating_margin);
  $('per').textContent=num(v.per);$('forwardPer').textContent=num(v.forward_per);$('pbr').textContent=num(v.pbr);$('psr').textContent=num(v.psr);
  $('cash').textContent=money(h.total_cash);$('debt').textContent=money(h.total_debt);$('debtEquity').textContent=num(h.debt_to_equity);$('currentRatio').textContent=num(h.current_ratio);
  $('dividendYield').textContent=pct(r.dividend_yield);$('dividendRate').textContent=num(r.dividend_rate);$('payoutRatio').textContent=pct(r.payout_ratio);
  const list=(id,a)=>$(id).innerHTML=(a||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li>該当項目なし</li>';
  list('fundGood',f.good);list('fundWarning',f.warning);list('fundBad',f.bad);
}
function renderCharts(rows){
  const labels=rows.map(x=>x.date); const common={};
  line('priceChart',labels,[ds('株価',rows.map(x=>x.close),'#62a6ff'),ds('MA20',rows.map(x=>x.ma20),'#ff6b8a'),ds('MA50',rows.map(x=>x.ma50),'#f5b942'),ds('MA200',rows.map(x=>x.ma200),'#a78bfa')]);
  line('volumeChart',labels,[ds('出来高',rows.map(x=>x.volume),'#4dd8c8'),ds('20日平均',rows.map(x=>x.volume_ratio!=null&&x.volume!=null?x.volume/x.volume_ratio:null),'#6d7b92',[5,4])]);
  line('bbChart',labels,[ds('株価',rows.map(x=>x.close),'#62a6ff'),ds('Upper',rows.map(x=>x.bb_upper),'#8b78ff',[5,4]),ds('Middle',rows.map(x=>x.bb_middle),'#f5b942'),ds('Lower',rows.map(x=>x.bb_lower),'#8b78ff',[5,4])]);
  line('rsiChart',labels,[ds('RSI',rows.map(x=>x.rsi),'#4dd8c8')],{scales:{y:{min:0,max:100,grid:{color:'rgba(255,255,255,.055)'}}}});
  line('macdChart',labels,[ds('MACD',rows.map(x=>x.macd),'#62a6ff'),ds('Signal',rows.map(x=>x.signal),'#ff6b8a'),ds('Histogram',rows.map(x=>x.macd_hist),'#a78bfa')]);
  line('stochChart',labels,[ds('Stochastic',rows.map(x=>x.stochastic),'#f5b942')],{scales:{y:{min:0,max:100,grid:{color:'rgba(255,255,255,.055)'}}}});
  line('rocChart',labels,[ds('ROC',rows.map(x=>x.roc),'#a78bfa')]);
  line('atrChart',labels,[ds('ATR',rows.map(x=>x.atr),'#ff9f5a')]);
  line('stdChart',labels,[ds('STD',rows.map(x=>x.std),'#62a6ff')]);
}
async function analyze(){
  const ticker=$('ticker').value.trim(); if(!ticker)return;
  const period=$('period').value; $('analyzeBtn').disabled=true;$('analyzeBtn').innerHTML='分析中…';
  try{ const r=await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}`); const data=await r.json(); if(!r.ok)throw new Error(data.error||'取得エラー'); lastData=data; renderMetric(data); window.scrollTo({top:280,behavior:'smooth'}); }
  catch(e){$('emptyState').classList.remove('hidden');$('emptyState').innerHTML=`<div class="error-icon">!</div><h2>データを取得できませんでした</h2><p>${esc(e.message)}</p>`;}
  finally{$('analyzeBtn').disabled=false;$('analyzeBtn').innerHTML='分析する <span>↗</span>';}
}

function getWatch(){try{return JSON.parse(localStorage.getItem('chartai_watchlist')||'[]')}catch{return[]}}
function setWatch(a){localStorage.setItem('chartai_watchlist',JSON.stringify([...new Set(a)].slice(0,50)));}
function updateWatchButton(t){const a=getWatch();$('watchBtn').textContent=a.includes(t)?'★ ウォッチ中':'☆ ウォッチ';$('watchBtn').classList.toggle('active',a.includes(t));}
$('watchBtn').addEventListener('click',()=>{if(!lastData)return;let a=getWatch();const t=lastData.ticker;a=a.includes(t)?a.filter(x=>x!==t):[...a,t];setWatch(a);updateWatchButton(t);});
async function renderWatchlist(){const box=$('watchlistGrid'),a=getWatch(); if(!a.length){box.innerHTML='<div class="empty-state compact"><div class="empty-orbit">☆</div><h2>ウォッチリストは空です</h2><p>分析画面の「ウォッチ」を押すとここに保存されます。</p></div>';return;} box.innerHTML=a.map(t=>`<button class="watch-item" data-ticker="${esc(t)}"><b>${esc(t)}</b><span>分析を開く ↗</span></button>`).join('');box.querySelectorAll('.watch-item').forEach(b=>b.addEventListener('click',()=>{$('ticker').value=b.dataset.ticker;showView('analysis');analyze();}));}

const filterKeys = ['per','pbr','peg','roe','roa','ebitda_margin','revenue_growth','eps_growth','dividend_yield','debt_equity','current_ratio','market_cap','price_change','volume'];
const filterLabels = {per:'PER',pbr:'PBR',peg:'PEG',roe:'ROE',roa:'ROA',ebitda_margin:'EBITDA率',revenue_growth:'売上成長率',eps_growth:'EPS成長率',dividend_yield:'配当利回り',debt_equity:'D/E',current_ratio:'流動比率',market_cap:'時価総額',price_change:'騰落率',volume:'出来高'};
const presets = {
  value: {label:'割安な銘柄', filters:{per:[0,15],pbr:[0,2]}, sort:'pe'},
  growth: {label:'成長している銘柄', filters:{revenue_growth:[10,null],eps_growth:[10,null]}, sort:'change'},
  dividend: {label:'配当を重視', filters:{dividend_yield:[3,null]}, sort:'marketcap'},
  quality: {label:'収益性を重視', filters:{roe:[10,null],roa:[5,null]}, sort:'marketcap'},
  large: {label:'大型株', filters:{market_cap:[100000000000,null]}, sort:'marketcap'},
  momentum: {label:'最近強い銘柄', filters:{price_change:[3,null],volume:[1000000,null]}, sort:'change'}
};

function setFilter(key, min, max, enabled=true){
  const cb=document.querySelector(`.filter-enable[data-key="${key}"]`);
  if(!cb) return;
  cb.checked=enabled;
  ['min','max'].forEach(bound=>{
    const input=document.querySelector(`.filter-input[data-key="${key}"][data-bound="${bound}"]`);
    if(input) input.value=bound==='min'?(min??''):(max??'');
  });
}
function clearFilters(){
  document.querySelectorAll('.filter-enable').forEach(x=>x.checked=false);
  document.querySelectorAll('.filter-input').forEach(x=>x.value='');
  document.querySelectorAll('.goal-card').forEach(x=>x.classList.remove('selected'));
  syncFilterInputs();
}
function applyPreset(name){
  const preset=presets[name]; if(!preset) return;
  clearFilters();
  Object.entries(preset.filters).forEach(([key,range])=>setFilter(key,range[0],range[1]));
  if($('screenMode')) $('screenMode').value=preset.sort;
  const card=document.querySelector(`.goal-card[data-preset="${name}"]`); if(card) card.classList.add('selected');
  syncFilterInputs();
  $('screenStatus').textContent=`「${preset.label}」の条件を設定しました。検索ボタンで実行できます。`;
}
function syncFilterInputs(){
  document.querySelectorAll('.filter-enable').forEach(cb=>{
    const key=cb.dataset.key;
    document.querySelectorAll(`.filter-input[data-key="${key}"]`).forEach(input=>{
      input.disabled=!cb.checked;
      input.closest('.filter-item')?.classList.toggle('enabled',cb.checked);
    });
  });
  updateFilterCount();
  renderActiveFilters();
}
function updateFilterCount(){
  const count=[...document.querySelectorAll('.filter-enable')].filter(x=>x.checked).length;
  $('filterCount').textContent=`${count}条件`;
}
function prettyFilter(key,bound,value){
  const n=Number(value); if(!Number.isFinite(n)) return '';
  if(key==='market_cap'){
    if(n>=1e12) return `${(n/1e12).toFixed(1)}兆円`;
    if(n>=1e8) return `${(n/1e8).toFixed(0)}億円`;
    return `${n.toLocaleString()}円`;
  }
  if(key==='volume') return n>=1e6?`${(n/1e6).toFixed(1)}百万株`:n.toLocaleString();
  return `${n}${['roe','roa','ebitda_margin','revenue_growth','eps_growth','dividend_yield','price_change'].includes(key)?'%':''}`;
}
function renderActiveFilters(){
  const box=$('activeFilterChips'); if(!box) return;
  const chips=[];
  document.querySelectorAll('.filter-enable:checked').forEach(cb=>{
    const key=cb.dataset.key;
    const min=document.querySelector(`.filter-input[data-key="${key}"][data-bound="min"]`)?.value;
    const max=document.querySelector(`.filter-input[data-key="${key}"][data-bound="max"]`)?.value;
    const range=min&&max?`${prettyFilter(key,'min',min)}〜${prettyFilter(key,'max',max)}`:min?`${prettyFilter(key,'min',min)}以上`:max?`${prettyFilter(key,'max',max)}以下`:'指定なし';
    chips.push(`<button class="active-chip" data-remove-filter="${key}">${esc(filterLabels[key])} <b>${esc(range)}</b> ×</button>`);
  });
  box.innerHTML=chips.join('')||'<span class="no-filter">条件なし（全体から探します）</span>';
  box.querySelectorAll('[data-remove-filter]').forEach(btn=>btn.addEventListener('click',()=>{setFilter(btn.dataset.removeFilter,'','',false);syncFilterInputs();}));
}
function buildScreenParams(){
  const p=new URLSearchParams({region:activeRegion,mode:$('screenMode').value,page:screenPage,size:100});
  document.querySelectorAll('.filter-enable:checked').forEach(cb=>{
    const key=cb.dataset.key;
    ['min','max'].forEach(bound=>{
      const input=document.querySelector(`.filter-input[data-key="${key}"][data-bound="${bound}"]`);
      if(input && input.value!=='') p.set(`${key}_${bound}`,input.value);
    });
  });
  return p;
}
async function loadScreener(){
  $('screenBtn').disabled=true;
  $('screenStatus').textContent='銘柄を探しています…';
  $('screenBody').innerHTML='<tr><td colspan="11" class="table-empty"><div class="loading-dot"></div> 条件に合う銘柄を取得しています…</td></tr>';
  const params=buildScreenParams();
  try{
    const r=await fetch(`/api/screener?${params.toString()}`); const data=await r.json();
    if(!r.ok) throw new Error(data.error||'取得エラー');
    $('screenTitle').textContent=activeRegion==='jp'?'日本株':'米国株';
    $('screenStatus').textContent=`${data.results?.length||0}件 · ページ ${screenPage+1} · ${data.filter_count||0}条件`;
    $('pageLabel').textContent=screenPage+1;
    $('screenBody').innerHTML=(data.results||[]).map(x=>`<tr><td><b>${esc(x.symbol)}</b><small>${esc(x.exchange)}</small></td><td>${esc(x.name)}</td><td>${num(x.price)}</td><td class="${x.change>=0?'positive':'negative'}">${x.change>=0?'+':''}${num(x.change)}%</td><td>${money(x.market_cap)}</td><td>${money(x.volume)}</td><td>${num(x.pe)}</td><td>${num(x.pbr)}</td><td>${pct(x.roe)}</td><td>${pct(x.dividend_yield)}</td><td><button class="mini-open" data-symbol="${esc(x.symbol)}">分析</button></td></tr>`).join('')||'<tr><td colspan="11" class="table-empty">条件に合う銘柄がありません。条件を少しゆるめて再検索してみてください。</td></tr>';
    document.querySelectorAll('.mini-open').forEach(b=>b.addEventListener('click',()=>{$('ticker').value=b.dataset.symbol;showView('analysis');analyze();}));
  }catch(e){
    $('screenBody').innerHTML=`<tr><td colspan="11" class="table-empty"><b>スクリーナーを取得できませんでした</b><br><small>${esc(e.message)}</small></td></tr>`;
    $('screenStatus').textContent='取得失敗';
  }finally{$('screenBtn').disabled=false;}
}

document.querySelectorAll('.filter-enable').forEach(cb=>cb.addEventListener('change',()=>{document.querySelectorAll('.goal-card').forEach(x=>x.classList.remove('selected'));syncFilterInputs();}));
document.querySelectorAll('.filter-input').forEach(input=>{
  input.addEventListener('input',()=>{document.querySelectorAll('.goal-card').forEach(x=>x.classList.remove('selected'));syncFilterInputs();});
  input.addEventListener('keydown',e=>{if(e.key==='Enter'){screenPage=0;loadScreener();}});
});
document.querySelectorAll('.goal-card').forEach(card=>card.addEventListener('click',()=>applyPreset(card.dataset.preset)));
document.querySelectorAll('.screen-level').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.screen-level').forEach(x=>x.classList.remove('active')); btn.classList.add('active');
  const easy=btn.dataset.level==='easy';
  $('easyScreen').classList.toggle('hidden',!easy);
  $('advancedScreen').classList.toggle('visible',!easy);
}));
document.querySelectorAll('.region').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.region').forEach(x=>x.classList.remove('active'));b.classList.add('active');activeRegion=b.dataset.region;screenPage=0;loadScreener();}));
$('screenBtn').addEventListener('click',()=>{screenPage=0;loadScreener();});
$('screenMode').addEventListener('change',()=>{screenPage=0;loadScreener();});
$('screenClear').addEventListener('click',()=>{clearFilters();$('screenStatus').textContent='条件をリセットしました。目的を選ぶか、詳細条件を設定してください。';$('screenBody').innerHTML='<tr><td colspan="11" class="table-empty">目的を選ぶか、詳細条件を設定して検索してください。</td></tr>';screenPage=0;});
$('prevPage').addEventListener('click',()=>{if(screenPage>0){screenPage--;loadScreener();}});
$('nextPage').addEventListener('click',()=>{screenPage++;loadScreener();});
syncFilterInputs();
