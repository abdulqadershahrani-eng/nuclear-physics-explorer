"""AI-Powered 3D Nuclear Physics Explorer V5.4 — full IAEA import, optimized 3D, transparent ML."""
from pathlib import Path
from io import BytesIO
import math
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parent
IAEA_URL='https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all'
CACHE=ROOT/'iaea_ground_states.csv'
SAMPLE=ROOT/'sample_isotopes.csv'
M_H=1.00782503223; M_N=1.00866491595; U_MEV=931.49410242
st.set_page_config(page_title='Nuclear Physics Explorer V5.4',page_icon='⚛️',layout='wide')
st.markdown("""<style>
.stApp{background:linear-gradient(125deg,#091326,#11203a 65%,#09152a);color:#eef5ff}
/* Contrasting slate-blue Streamlit header. */
[data-testid="stHeader"], header[data-testid="stHeader"]{background:#30496c!important;color:#eef5ff!important;border-bottom:1px solid #54749a!important;}
[data-testid="stToolbar"]{background:transparent!important;}
/* Hide only the Streamlit Deploy control; retain the menu and status indicators. */
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stDeployButton, button[kind="headerNoPadding"]:has([data-testid="stDeployButton"]){display:none!important;}
[data-testid="stSidebar"]{background:#101c31}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#edf5ff!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#bfd2ef!important;font-size:.88rem!important;line-height:1.5!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="input"]>div{background:#f7fbff!important;color:#13233c!important;border-color:#9eb7d5!important}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] svg{color:#13233c!important;fill:#13233c!important}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="popover"] [role="option"]{background:#f7fbff!important;color:#13233c!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{border:1px solid #5a759a;border-radius:10px;padding:6px}
[data-testid="stSidebar"] [data-testid="stAlert"] p{color:#143052!important}
[data-testid="stSidebar"] button{font-weight:650}
.block-container{max-width:1750px;padding-top:1.5rem}
h1,h2,h3{color:#f5f9ff!important}
[data-testid="stMetric"]{background:#192c49;border:1px solid #344966;border-radius:12px;padding:12px}
[data-testid="stMetricLabel"] *{color:#bcd2ed!important}
[data-testid="stMetricValue"]{color:#fff!important}
[data-testid="stCaptionContainer"] *{color:#b6c9e5!important}
button[data-baseweb="tab"]{color:#dbe9ff!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#ff7086!important}
/* Accessible contrast on inputs and chart controls */
[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#f7fbff!important;color:#15243c!important}
[data-testid="stSelectbox"] [data-baseweb="select"] span{color:#15243c!important}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def parse_iaea(content):
    raw=pd.read_csv(BytesIO(content),low_memory=False)
    raw.columns=raw.columns.astype(str).str.strip().str.lower()
    required={'z','n','symbol','atomic_mass','half_life_sec','abundance'}
    missing=required-set(raw.columns)
    if missing: raise ValueError('Missing IAEA columns: '+', '.join(sorted(missing)))
    for col in ['z','n','atomic_mass','half_life_sec','abundance']:
        raw[col]=pd.to_numeric(raw[col],errors='coerce')
    raw=raw.loc[raw.z.notna() & raw.n.notna() & (raw.z>=1) & (raw.n>=0)].copy()
    raw['Z']=raw.z.astype(int);raw['N']=raw.n.astype(int);raw['A']=raw.Z+raw.N
    raw['isotope']=raw.symbol.astype(str).str.strip().str.capitalize()+'-'+raw.A.astype(str)
    raw['atomic_mass_u']=raw.atomic_mass/1_000_000.0
    raw.loc[~np.isfinite(raw.atomic_mass_u) | (raw.atomic_mass_u<=0),'atomic_mass_u']=np.nan
    # Operational classroom labels: finite positive half-life -> radioactive;
    # otherwise positive natural abundance -> stable; all other rows remain unknown.
    decay=raw.half_life_sec.notna() & (raw.half_life_sec>0)
    abundance=raw.abundance.notna() & (raw.abundance>0)
    raw['stable']=pd.Series(pd.NA,index=raw.index,dtype='Int64')
    raw.loc[decay,'stable']=0
    raw.loc[~decay & abundance,'stable']=1
    raw=raw.drop_duplicates(['Z','N']).sort_values(['Z','A']).reset_index(drop=True)
    # Preserve available LiveChart metadata for the dynamic isotope biography.
    optional = ['half_life_sec', 'abundance', 'decay_1', 'decay_1_%',
                'discovery', 'discovery_year', 'year', 'ensdf_publication_cut_off',
                'ensdf_publication', 'ensdf_authors', 'authors']
    retained = ['isotope','Z','N','A','atomic_mass_u','stable']
    for col in optional:
        if col in raw.columns and col not in retained:
            retained.append(col)
    return raw[retained],len(raw)

@st.cache_data(show_spinner=False)
def parse_sample(content):
    df=pd.read_csv(BytesIO(content))
    df['stable']=df.stable.astype('Int64')
    return df,len(df)

@st.cache_data(show_spinner=False,ttl=86400)
def fetch_iaea():
    response=requests.get(IAEA_URL,timeout=100,headers={'User-Agent':'Mozilla/5.0 (educational nuclear physics visualization)','Accept':'text/csv,*/*'})
    response.raise_for_status()
    content=response.content
    if b'z' not in content[:1000].lower(): raise ValueError('IAEA did not return a CSV. Use manual CSV upload instead.')
    parse_iaea(content)
    return content


# These short application summaries are editorial context, NOT LiveChart fields.
# Never synthesize isotope-specific historical or ENSDF bibliographic details.
APPLICATIONS = {
    'U-235': 'It is used as fissile fuel in nuclear reactors and is important in nuclear fission research.',
    'U-238': 'It is the principal isotope of natural uranium and can be converted into plutonium-239 in a reactor.',
    'C-14': 'It is used in radiocarbon dating of once-living materials.',
    'Co-60': 'Its gamma rays are used in medical-equipment sterilization and industrial radiography.',
    'Tc-99m': 'It is widely used as a radiotracer in diagnostic nuclear medicine.',
    'I-131': 'It is used in diagnosing and treating certain thyroid conditions.',
    'F-18': 'It is used in PET imaging, commonly in fluorodeoxyglucose (FDG).',
    'H-2': 'Deuterium is used in heavy water and fusion research.',
    'H-3': 'Tritium is used in fusion research and as a radioactive tracer.',
    'He-3': 'It is used in neutron detection and low-temperature physics.',
    'Fe-56': 'It is important in nuclear binding-energy studies and is the most abundant stable iron isotope.',
}
# Abundances below are approximate reference values from standard isotope knowledge,
# explicitly distinguished from dataset-provided abundance.
REFERENCE_ABUNDANCE = {
    'U-235': 'approximately 0.72% of natural uranium',
    'U-238': 'approximately 99.27% of natural uranium',
    'Fe-56': 'approximately 91.75% of natural iron',
}

def readable_half_life(seconds):
    try:
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0:
            return None
    except (TypeError, ValueError):
        return None
    year = 365.25 * 24 * 3600
    for unit, scale in [('billion years', year*1e9), ('million years', year*1e6),
                        ('years', year), ('days', 86400), ('hours', 3600),
                        ('minutes', 60), ('seconds', 1)]:
        if seconds >= scale:
            return f'{seconds/scale:,.3g} {unit}'
    return f'{seconds:.3g} seconds'

def available_value(record, *names):
    for name in names:
        if name in record.index:
            value = record[name]
            if pd.notna(value) and str(value).strip() and str(value).lower() not in ('nan', 'none'):
                return value
    return None

def isotope_biography(record):
    """Build a data-grounded biography for every nuclide, without fabricated history."""
    name = str(record['isotope'])
    z, n, a = int(record.Z), int(record.N), int(record.A)
    ratio = n / z if z else 0
    if pd.isna(record.stable):
        status = 'unknown under the app’s simplified stability-label rules'
    else:
        status = 'stable' if int(record.stable) else 'radioactive'
    sentences = [
        f'**{name}** has **{z} protons** and **{n} neutrons**, giving it a mass number '
        f'of **{a}** and a neutron-to-proton ratio of **{ratio:.2f}**. '
        f'The app’s IAEA-derived classroom classification is **{status}**.'
    ]
    if n > z:
        sentences.append('It contains more neutrons than protons. Extra neutrons help offset '
                         'electrostatic repulsion in many heavier nuclei, although the ratio alone '
                         'cannot determine nuclear stability.')
    elif n == z:
        sentences.append('It has equal numbers of protons and neutrons. This balance is common '
                         'among lighter stable nuclei, but does not by itself guarantee stability.')
    else:
        sentences.append('It has fewer neutrons than protons. Proton-rich nuclei can undergo '
                         'positron emission or electron capture when energetically allowed, '
                         'but the actual decay mode must be checked against measured data.')

    mass = available_value(record, 'atomic_mass_u')
    try:
        mass = float(mass)
        be = binding_energy(z, n, mass)
        if np.isfinite(be) and be > 0:
            per_nucleon = be/a
            sentences.append(
                f'Using its tabulated atomic mass, the calculated binding energy is '
                f'**{be:,.2f} MeV** (**{per_nucleon:.3f} MeV per nucleon**). '
                'This is the energy required to separate the nucleus into free nucleons; '
                'it is not a direct measurement of radioactive half-life.')
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    half = readable_half_life(available_value(record, 'half_life_sec'))
    if half:
        sentences.append(f'The IAEA ground-state data give a half-life of approximately **{half}**.')
    elif status == 'stable':
        sentences.append('No radioactive half-life is reported for this stable label.')
    else:
        sentences.append('A numerical half-life is not available in the loaded record.')

    decay = available_value(record, 'decay_1')
    if decay:
        sentences.append(f'The dataset lists **{str(decay).strip()}** as a decay mode.')

    abundance = available_value(record, 'abundance')
    try:
        abundance = float(abundance)
        if not np.isfinite(abundance) or abundance <= 0:
            abundance = None
    except (TypeError, ValueError):
        abundance = None
    if abundance is not None:
        sentences.append(f'Its reported natural isotopic abundance is **{abundance:.5g}%**.')
    elif name in REFERENCE_ABUNDANCE:
        sentences.append(f'As supplemental reference context, it accounts for '
                         f'**{REFERENCE_ABUNDANCE[name]}**.')
    else:
        sentences.append('The loaded data do not establish a numerical natural abundance '
                         'for this isotope; this should not be interpreted as zero abundance.')

    if name in APPLICATIONS:
        sentences.append('**Known application:** ' + APPLICATIONS[name])
    elif status == 'radioactive':
        sentences.append('**Scientific relevance:** Radioactive isotopes can be investigated '
                         'for tracer studies, radiation detection, or decay research. '
                         'This is a general research context, not a claim that this particular '
                         'isotope has an established practical use.')
    else:
        sentences.append('**Scientific relevance:** Stable-isotope composition is relevant '
                         'to nuclear-mass measurements and isotopic analysis. '
                         'No isotope-specific commercial application has been verified here.')

    year = available_value(record, 'discovery_year', 'discovery', 'year')
    if year is not None:
        try:
            year_num = float(year)
            year_text = str(int(year_num)) if year_num.is_integer() else str(year)
        except (TypeError, ValueError):
            year_text = str(year)
        sentences.append(f'The loaded IAEA record includes discovery information: **{year_text}**.')
    else:
        sentences.append('A verified discovery year is not provided in the loaded record.')

    publication = available_value(record, 'ensdf_publication', 'ensdf_publication_cut_off')
    authors = available_value(record, 'ensdf_authors', 'authors')
    if publication is not None:
        sentences.append(f'ENSDF publication information: **{publication}**.')
    if authors is not None:
        sentences.append(f'ENSDF author information: **{authors}**.')
    if publication is None and authors is None:
        sentences.append('ENSDF publication and author details are not included in the '
                         'loaded data; consult the ENSDF entry for bibliographic details.')
    return ' '.join(sentences)

# Automatically load a saved IAEA export, or retrieve it once when online.
# The sample remains an explicitly labeled fallback, never represented as full IAEA data.
@st.cache_data(show_spinner=False)
def auto_download_once():
    try:
        return fetch_iaea(), None
    except Exception as exc:
        return None, str(exc)

st.sidebar.header('Nucleus controls')
with st.sidebar.expander('Advanced settings · IAEA data', expanded=False):
    st.caption('The official IAEA ground-state dataset loads automatically. Manual upload is optional.')
    upload = st.file_uploader('Replace data with IAEA ground_states CSV', type=['csv'])
    force_refresh = st.button('Refresh IAEA dataset', use_container_width=True)
    st.markdown(f'[Official IAEA CSV download]({IAEA_URL})')

error_message = None
if upload is not None:
    content = upload.getvalue()
    source = 'Manually uploaded IAEA LiveChart CSV'
    full = True
elif force_refresh:
    with st.spinner('Refreshing IAEA ground-state data...'):
        try:
            content = fetch_iaea()
            CACHE.write_bytes(content)
            source = 'Official IAEA LiveChart CSV (refreshed and saved)'
            full = True
        except Exception as exc:
            error_message = str(exc)
            if CACHE.exists():
                content = CACHE.read_bytes()
                source = 'Previously saved IAEA LiveChart CSV'
                full = True
            else:
                content = SAMPLE.read_bytes()
                source = 'Bundled DEMO sample (not full IAEA)'
                full = False
elif CACHE.exists():
    content = CACHE.read_bytes()
    source = 'Saved official IAEA LiveChart CSV (offline-ready)'
    full = True
else:
    with st.spinner('First launch: automatically downloading IAEA ground-state data...'):
        downloaded, error_message = auto_download_once()
    if downloaded is not None:
        content = downloaded
        CACHE.write_bytes(content)
        source = 'Official IAEA LiveChart CSV (automatically downloaded and saved)'
        full = True
    else:
        content = SAMPLE.read_bytes()
        source = 'Bundled DEMO sample (not full IAEA)'
        full = False

try:
    df,total = parse_iaea(content) if full else parse_sample(content)
except Exception as exc:
    st.error(f'Cannot read nuclear dataset: {exc}')
    st.stop()
if not full:
    st.warning('DEMO MODE: the full IAEA dataset could not be downloaded automatically. '
               'Connect to the internet and click Refresh IAEA dataset under Advanced settings, '
               'or upload the official CSV there. The sample is NOT the full IAEA collection.')
    if error_message:
        st.caption('Automatic download error: ' + error_message)
elif error_message:
    st.info('IAEA refresh failed, so the previously saved complete dataset is being used.')

mass_ok=df.atomic_mass_u.notna() & np.isfinite(df.atomic_mass_u)
label_ok=df.stable.notna()
training=df.loc[mass_ok & label_ok].copy().reset_index(drop=True)
if len(training)<30 or training.stable.nunique()!=2:
    st.error('Insufficient known stable/radioactive labels for ML training. Check the imported IAEA file.');st.stop()
training['stable']=training.stable.astype(int)

# Nuclear mass is a separate physics calculation; it is NOT inferred by the ML model.
def binding_energy(z,n,mass):
    return (z*M_H+n*M_N-mass)*U_MEV

def features(frame):
    z=frame.Z.to_numpy(dtype=float);n=frame.N.to_numpy(dtype=float)
    return pd.DataFrame({'Z':z,'N':n,'A':z+n,'N/Z':n/np.maximum(z,1),
      'asymmetry':(n-z)/np.maximum(n+z,1),'evenZ':(z%2==0).astype(int),'evenN':(n%2==0).astype(int)})

@st.cache_resource(show_spinner=False)
def fit_model(csv_bytes):
    data=pd.read_csv(BytesIO(csv_bytes))
    data.stable=data.stable.astype(int)
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=19)
    train,test=next(split.split(data,groups=data.Z))
    args=dict(n_estimators=250,max_depth=12,min_samples_leaf=2,class_weight='balanced_subsample',random_state=19,n_jobs=-1)
    eval_model=RandomForestClassifier(**args).fit(features(data.iloc[train]),data.stable.iloc[train])
    truth=data.stable.iloc[test];pred=eval_model.predict(features(data.iloc[test]))
    baseline=np.full(len(test),int(data.stable.iloc[train].mode().iloc[0]))
    precision,recall,f1,_=precision_recall_fscore_support(truth,pred,labels=[0,1],zero_division=0)
    stats={'accuracy':accuracy_score(truth,pred),'balanced':balanced_accuracy_score(truth,pred),
      'baseline':accuracy_score(truth,baseline),'n_train':len(train),'n_test':len(test),
      'test_z':sorted(data.Z.iloc[test].unique().tolist()),'confusion':confusion_matrix(truth,pred,labels=[0,1]).tolist(),
      'precision':precision[1],'recall':recall[1],'f1':f1[1]}
    deployed=RandomForestClassifier(**args).fit(features(data),data.stable)
    return deployed,stats

with st.spinner('Training the Random Forest model...'):
    model,stats=fit_model(training.to_csv(index=False).encode())

@st.cache_data(show_spinner=False)
def geometry(res):
    # Low-polygon meshes preserve individual nucleons while keeping heavy nuclei interactive.
    lat=np.linspace(.001,math.pi-.001,res)
    lon=np.linspace(0,2*math.pi,2*res,endpoint=False)
    x=np.outer(np.sin(lat),np.cos(lon)).ravel()
    y=np.outer(np.sin(lat),np.sin(lon)).ravel()
    z=np.outer(np.cos(lat),np.ones_like(lon)).ravel()
    i=[];j=[];k=[];w=len(lon)
    for a in range(res-1):
        for b in range(w):
            c=a*w+b;d=a*w+(b+1)%w;e=(a+1)*w+b;f=(a+1)*w+(b+1)%w
            i.extend((c,d));j.extend((e,e));k.extend((d,f))
    return x,y,z,np.asarray(i),np.asarray(j),np.asarray(k)

def nucleus_figure(z,n,gloss):
    count=z+n
    # Deterministic, compact, volume-distributed schematic positions.
    idx=np.arange(count,dtype=float)+.5
    golden=math.pi*(3-math.sqrt(5))*idx
    h=1-2*idx/count
    planar=np.sqrt(np.maximum(0,1-h*h))
    radius=0.15+0.85*np.cbrt(idx/count)
    pos=np.column_stack((planar*np.cos(golden),planar*np.sin(golden),h))*radius[:,None]
    # Use constant nucleon radius scaled to avoid huge overlaps for heavy nuclei.
    scale=0.5*count**(1/3)
    pos*=scale
    ball=0.47
    resolution=11 if count<=80 else (8 if count<=170 else 6)
    sx,sy,sz,ii,jj,kk=geometry(resolution)
    labels=np.array([1]*z+[0]*n)
    np.random.default_rng(1000+101*z+n).shuffle(labels)
    fig=go.Figure()
    for p,kind in zip(pos,labels):
        color='#ff5c70' if kind else '#3aaaff'
        fig.add_trace(go.Mesh3d(x=p[0]+ball*sx,y=p[1]+ball*sy,z=p[2]+ball*sz,
          i=ii,j=jj,k=kk,color=color,opacity=1,flatshading=False,
          lighting=dict(ambient=.35 if gloss else .75,diffuse=.85 if gloss else .4,
                        specular=.95 if gloss else .08,roughness=.16 if gloss else .85,fresnel=.2),
          lightposition=dict(x=70,y=55,z=110),showlegend=False,
          hovertemplate=('Proton' if kind else 'Neutron')+'<extra></extra>'))
    limit=max(2.4,scale+1.1)
    fig.update_layout(height=560,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='#09152b',plot_bgcolor='#09152b',
      scene=dict(bgcolor='#09152b',xaxis=dict(visible=False,range=[-limit,limit]),
       yaxis=dict(visible=False,range=[-limit,limit]),zaxis=dict(visible=False,range=[-limit,limit]),
       aspectmode='cube',camera=dict(eye=dict(x=1.5,y=1.25,z=.95))),
      dragmode='orbit',showlegend=False,uirevision='nucleus')
    return fig

st.title('⚛️ AI-Powered 3D Nuclear Physics Explorer')

st.sidebar.header('Nucleus controls')
choices=df.isotope.tolist()
default=choices.index('U-235') if 'U-235' in choices else (choices.index('Ne-22') if 'Ne-22' in choices else 0)
choice=st.sidebar.selectbox('Choose isotope',choices,index=default)
row=df.loc[df.isotope==choice].iloc[0]
z=int(row.Z);n=int(row.N);a=z+n
st.sidebar.markdown(f'**Selected: {choice}**  \n🔴 Protons (Z): **{z}**  \n🔵 Neutrons (N): **{n}**  \nMass number (A = Z + N): **{a}**')
auto=st.sidebar.toggle('Automatic rotation',value=True)
gloss=st.sidebar.toggle('Glossy lighting',value=True)
speed=st.sidebar.slider('Rotation speed',.2,2.0,.65,.05,disabled=not auto)
st.sidebar.markdown('🔴 Protons  ·  🔵 Neutrons')
st.sidebar.info('**3D controls**\n\nDrag = rotate manually\n\nMouse wheel = zoom\n\nAutomatic rotation resumes after interaction.')
st.sidebar.markdown(
    '<div style="margin-top:26px;padding:12px 10px;border-top:1px solid #526a89;'
    'font-size:0.85rem;line-height:1.7;color:#c7d8ef">'
    '<strong style="color:#f0f6ff">Contact</strong><br>'
    '<a href="mailto:g202516670@kfupm.edu.sa" style="color:#9cc9ff;'
    'text-decoration:underline;overflow-wrap:anywhere">g202516670@kfupm.edu.sa</a>'
    '</div>', unsafe_allow_html=True
)

left,right=st.columns([1.55,1],gap='large')
with left:
    st.subheader(f'Interactive 3D nucleus — {choice}')
    fig=nucleus_figure(z,n,gloss)
    html=pio.to_html(fig,include_plotlyjs=True,full_html=True,config={'displaylogo':False,'responsive':True})
    js=f'''<script>(function(){{const el=document.querySelector('.plotly-graph-div');if(!el)return;
    const enabled={str(auto).lower()},speed={speed:.3f};if(!enabled)return;
    let angle=Math.atan2(1.25,1.5),last=performance.now(),pause=0,drag=false,busy=false;
    const radius=Math.hypot(1.5,1.25),elev=.95;
    el.addEventListener('pointerdown',()=>{{drag=true;pause=performance.now()+3500;}},true);
    window.addEventListener('pointerup',()=>{{drag=false;pause=performance.now()+3500;}},true);
    el.addEventListener('wheel',()=>{{pause=performance.now()+3500;}},{{passive:true}});
    function tick(now){{const dt=Math.min((now-last)/1000,.1);last=now;
    if(now>pause&&!drag&&!busy&&document.visibilityState==='visible'){{angle+=dt*speed*.35;busy=true;
    Plotly.relayout(el,{{'scene.camera.eye':{{x:radius*Math.cos(angle),y:radius*Math.sin(angle),z:elev}}}})
    .catch(()=>{{}}).finally(()=>{{busy=false;}});}}
    requestAnimationFrame(tick);}}requestAnimationFrame(tick);}})();</script>'''
    components.html(html.replace('</body>',js+'</body>'),height=565,scrolling=False)
    st.caption('Each sphere represents one proton or neutron. 3D geometry is illustrative, not a nuclear quantum-mechanical simulation.')
    st.markdown('<div style="background:#192c49;border:1px solid #344966;border-radius:12px;padding:12px 16px;margin-top:10px"><span style="font-weight:700;font-size:1.05rem;color:#f5f9ff">📖 About This Isotope</span></div>', unsafe_allow_html=True)
    st.markdown(isotope_biography(row))
    st.caption('This paragraph updates for every isotope. IAEA fields are distinguished from supplementary examples and general physics explanations. Unknown historical, abundance, and application facts are not presented as established facts.')
with right:
    st.subheader('Nuclear analysis')
    predicted=int(model.predict(features(pd.DataFrame([row])))[0])
    vote=float(model.predict_proba(features(pd.DataFrame([row])))[0][list(model.classes_).index(predicted)])
    c1,c2=st.columns(2)
    c1.metric('AI prediction','Stable' if predicted else 'Radioactive')
    c2.metric('IAEA-derived label','Unknown' if pd.isna(row.stable) else ('Stable' if int(row.stable) else 'Radioactive'))
    st.progress(vote,text=f'Random Forest vote fraction: {vote:.0%} (not a calibrated probability)')
    st.caption(f'AI model trained on {len(training):,} labeled records. The prediction is a Random Forest classification; the IAEA-derived label is a separate reference obtained from simplified rules. Predictions for unknown labels are not validated.')
    if pd.notna(row.atomic_mass_u):
        be=binding_energy(z,n,float(row.atomic_mass_u))
        if be>=0:
            c1,c2=st.columns(2);c1.metric('Binding energy',f'{be:,.3f} MeV');c2.metric('BE per nucleon',f'{be/a:.3f} MeV')
        else: st.warning('The imported mass yields a negative binding energy; check the data before interpreting it.')
        st.caption(f'Atomic mass: {float(row.atomic_mass_u):.9f} u')
    else: st.info('No atomic mass reported for this nuclide: binding energy cannot be calculated.')
    st.metric('Neutron / proton ratio',f'{n/z:.3f}')
    st.latex(r'BE=[Zm(^1H)+Nm_n-m_{atom}(Z,N)]c^2')
    st.caption('Physics calculation using atomic masses, 1 u c² = 931.494 MeV; small electron-binding corrections neglected. Not an AI output.')

st.divider()
tab1,tab2,tab4=st.tabs(['Compare isotopes','Binding-energy curve','Dataset & limitations'])
with tab1:
    other=st.selectbox('Compare against',choices,index=choices.index('Fe-56') if 'Fe-56' in choices else 0)
    compare=df.loc[df.isotope.isin([choice,other]) & df.atomic_mass_u.notna()].copy()
    if not compare.empty:
        compare['BE/A']=compare.apply(lambda r:binding_energy(int(r.Z),int(r.N),float(r.atomic_mass_u))/int(r.A),axis=1)
        bar=go.Figure(go.Bar(x=compare.isotope,y=compare['BE/A'],text=compare['BE/A'].round(3),textposition='outside',marker_color=['#43adff' if v==choice else '#ff6676' for v in compare.isotope]))
        bar.update_layout(height=320,margin=dict(l=25,r=20,t=30,b=30),paper_bgcolor='#101d34',plot_bgcolor='#101d34',font_color='#eaf1ff',yaxis_title='MeV per nucleon')
        st.plotly_chart(bar,use_container_width=True)
    else: st.info('Atomic masses are missing for the selected comparison.')
with tab2:
    curve=df.loc[df.atomic_mass_u.notna()].copy()
    curve['BE/A']=(curve.Z*M_H+curve.N*M_N-curve.atomic_mass_u)*U_MEV/curve.A
    curve=curve.loc[(curve['BE/A']>=0)&(curve['BE/A']<12)]
    fig2=go.Figure()
    fig2.add_trace(go.Scattergl(x=curve.A,y=curve['BE/A'],mode='markers',name='IAEA nuclides',
       marker=dict(size=5,color='#4ab1ff',opacity=.55),customdata=curve[['isotope']].to_numpy(),hovertemplate='%{customdata[0]}<br>A=%{x}<br>BE/A=%{y:.3f} MeV<extra></extra>'))
    if pd.notna(row.atomic_mass_u):
        fig2.add_trace(go.Scatter(x=[a],y=[binding_energy(z,n,float(row.atomic_mass_u))/a],mode='markers+text',text=[choice],textposition='top center',marker=dict(size=13,color='#ff6676'),name='Selected isotope'))
    fig2.update_layout(height=470,paper_bgcolor='#101d34',plot_bgcolor='#101d34',font_color='#eaf1ff',xaxis_title='Mass number A',yaxis_title='Binding energy per nucleon (MeV)',legend=dict(orientation='h'))
    st.plotly_chart(fig2,use_container_width=True)
    st.caption('The curve shows why binding-energy-per-nucleon trends matter for fission and fusion. It does not by itself determine reaction feasibility or Q-values.')
with tab4:
    st.markdown(f'**Active source:** {source}. **Total imported:** {len(df):,}. **Z range:** {df.Z.min()}–{df.Z.max()}.')
    st.markdown(f'**IAEA official source:** [Download ground-state CSV]({IAEA_URL})')
    st.dataframe(df,height=320,use_container_width=True,hide_index=True)
    st.download_button('Download processed dataset',df.to_csv(index=False).encode(),file_name='nuclear_explorer_v5_4_processed.csv',mime='text/csv')
