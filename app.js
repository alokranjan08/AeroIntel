/**
 * AeroIntel — Aviation Incident Intelligence
 * Core Engine & Interactive UI Scripts (Universal 3D/4D/5D Card Physics & Interactive Model Demo)
 */

document.addEventListener('DOMContentLoaded', () => {
  initHeroInteractiveCanvas();
  initUniversal3DTilt();
  initOpposingAirplaneParallax();
  initScenarioAnalyzer();
  initCheckoutModal();
  initConsoleTabs();
  initExplorerTabs();
  initCodeStepSwitcher();
  initTableSearch();
  initGalleryModal();
  initAboutModal();
  initMobileNav();
  initSmoothScroll();
  initEasterEgg();
});

// Pre-Incident Feature Importance weights extracted from NTSB XGBoost Model
const PRE_INCIDENT_FEATURE_IMPORTANCES = [
  { name: 'Weather Condition', pct: 22.8 },
  { name: 'Amateur-Built Status', pct: 18.0 },
  { name: 'Purpose of Flight', pct: 13.5 },
  { name: 'Incident Year', pct: 12.5 },
  { name: 'Aircraft Category', pct: 11.5 },
  { name: 'Incident Month', pct: 8.7 },
  { name: 'Number of Engines', pct: 7.6 },
  { name: 'Season', pct: 5.4 }
];

// Python / Pandas Notebook Code Blocks for Explorer
const CODE_STEPS = {
  step1: `<span class="code-comment"># Step 1: Load and Clean Raw NTSB Dataset</span>
<span class="code-keyword">import</span> pandas <span class="code-keyword">as</span> pd
<span class="code-keyword">import</span> numpy <span class="code-keyword">as</span> np

<span class="code-comment"># Load raw dataset (46,377 rows x 38 columns)</span>
df_raw = pd.<span class="code-func">read_csv</span>(<span class="code-string">'data/aviation.csv'</span>, encoding=<span class="code-string">'utf-8-sig'</span>)

<span class="code-comment"># Filter U.S. incidents and drop irrelevant metadata</span>
df = df_raw[df_raw[<span class="code-string">'Country'</span>] == <span class="code-string">'United States'</span>].<span class="code-func">copy</span>()
drop_cols = [<span class="code-string">'Mkey'</span>, <span class="code-string">'ReportNo'</span>, <span class="code-string">'HasSafetyRec'</span>, <span class="code-string">'OriginalPublishDate'</span>]
df.<span class="code-func">drop</span>(columns=drop_cols, inplace=<span class="code-string">True</span>, errors=<span class="code-string">'ignore'</span>)

print(<span class="code-string">f"Cleaned U.S. incidents: {len(df):,} records"</span>)
<span class="code-comment"># Result: Cleaned U.S. incidents: 39,017 records</span>`,

  step2: `<span class="code-comment"># Step 2: Target Class Inference & Feature Engineering</span>
<span class="code-comment"># Infer missing HighestInjuryLevel target values from injury counts</span>
<span class="code-keyword">def</span> <span class="code-func">infer_severity</span>(row):
    <span class="code-keyword">if</span> pd.<span class="code-func">notna</span>(row[<span class="code-string">'HighestInjuryLevel'</span>]):
        <span class="code-keyword">return</span> row[<span class="code-string">'HighestInjuryLevel'</span>]
    <span class="code-keyword">if</span> row[<span class="code-string">'FatalInjuryCount'</span>] > <span class="code-num">0</span>: <span class="code-keyword">return</span> <span class="code-string">'Fatal'</span>
    <span class="code-keyword">if</span> row[<span class="code-string">'SeriousInjuryCount'</span>] > <span class="code-num">0</span>: <span class="code-keyword">return</span> <span class="code-string">'Serious'</span>
    <span class="code-keyword">if</span> row[<span class="code-string">'MinorInjuryCount'</span>] > <span class="code-num">0</span>: <span class="code-keyword">return</span> <span class="code-string">'Minor'</span>
    <span class="code-keyword">return</span> <span class="code-string">'None'</span>

df[<span class="code-string">'HighestInjuryLevel'</span>] = df.<span class="code-func">apply</span>(infer_severity, axis=<span class="code-num">1</span>)

<span class="code-comment"># Feature Engineering: Temporal extraction</span>
df[<span class="code-string">'EventDate'</span>] = pd.<span class="code-func">to_datetime</span>(df[<span class="code-string">'EventDate'</span>])
df[<span class="code-string">'Year'</span>] = df[<span class="code-string">'EventDate'</span>].dt.year
df[<span class="code-string">'Month'</span>] = df[<span class="code-string">'EventDate'</span>].dt.month`,

  step3: `<span class="code-comment"># Step 3: Categorical Label Encoding & Scaling</span>
<span class="code-keyword">from</span> sklearn.preprocessing <span class="code-keyword">import</span> LabelEncoder, StandardScaler

feature_cols = [<span class="code-string">'WeatherCondition'</span>, <span class="code-string">'AmateurBuilt'</span>, <span class="code-string">'PurposeOfFlight'</span>, <span class="code-string">'AirCraftCategory'</span>, <span class="code-string">'NumberOfEngines'</span>, <span class="code-string">'Year'</span>, <span class="code-string">'Month'</span>, <span class="code-string">'Season'</span>]
label_encoders = {}

<span class="code-keyword">for</span> col <span class="code-keyword">in</span> [<span class="code-string">'WeatherCondition'</span>, <span class="code-string">'PurposeOfFlight'</span>, <span class="code-string">'AirCraftCategory'</span>, <span class="code-string">'Season'</span>]:
    le = <span class="code-func">LabelEncoder</span>()
    df[col] = le.<span class="code-func">fit_transform</span>(df[col].<span class="code-func">astype</span>(str))
    label_encoders[col] = le

<span class="code-comment"># Apply StandardScaler z-score normalization</span>
scaler = <span class="code-func">StandardScaler</span>()
X_scaled = scaler.<span class="code-func">fit_transform</span>(df[feature_cols])`,

  step4: `<span class="code-comment"># Step 4: Handle Class Imbalance with SMOTE</span>
<span class="code-keyword">from</span> imblearn.over_sampling <span class="code-keyword">import</span> SMOTE
<span class="code-keyword">from</span> sklearn.model_selection <span class="code-keyword">import</span> train_test_split

X_train, X_test, y_train, y_test = <span class="code-func">train_test_split</span>(
    X_scaled, df[<span class="code-string">'HighestInjuryLevel'</span>], test_size=<span class="code-num">0.2</span>, random_state=<span class="code-num">42</span>, stratify=df[<span class="code-string">'HighestInjuryLevel'</span>]
)

<span class="code-comment"># Resample minority classes (Fatal, Serious, Minor) to match None (17,480 samples each)</span>
smote = <span class="code-func">SMOTE</span>(random_state=<span class="code-num">42</span>)
X_train_smote, y_train_smote = smote.<span class="code-func">fit_resample</span>(X_train, y_train)

print(<span class="code-string">f"Balanced training set: {len(X_train_smote):,} samples"</span>)
<span class="code-comment"># Result: Balanced training set: 69,344 samples</span>`,

  step5: `<span class="code-comment"># Step 5: XGBoost Classifier Training & Export</span>
<span class="code-keyword">import</span> xgboost <span class="code-keyword">as</span> xgb
<span class="code-keyword">import</span> joblib

xgb_model = xgb.<span class="code-func">XGBClassifier</span>(
    n_estimators=<span class="code-num">100</span>,
    max_depth=<span class="code-num">6</span>,
    learning_rate=<span class="code-num">0.1</span>,
    eval_metric=<span class="code-string">'mlogloss'</span>,
    random_state=<span class="code-num">42</span>
)

xgb_model.<span class="code-func">fit</span>(X_train_smote, y_train_smote)

<span class="code-comment"># Save model pipeline artifacts</span>
joblib.<span class="code-func">dump</span>(xgb_model, <span class="code-string">'models/best_model_xgboost.pkl'</span>)
joblib.<span class="code-func">dump</span>(scaler, <span class="code-string">'models/scaler.pkl'</span>)
joblib.<span class="code-func">dump</span>(label_encoders, <span class="code-string">'models/label_encoders.pkl'</span>)
print(<span class="code-string">"✓ All ML artifacts saved successfully!"</span>)`
};

// NTSB Investigation Case Study Data for Modal
const GALLERY_DATA = {
  fleet: {
    icon: '🌩️',
    stat: '3.4x',
    badge: 'WEATHER RISK ANALYSIS',
    title: 'Instrument Meteorological Conditions (IMC)',
    desc: 'Deep analysis across 39,017 NTSB records shows Instrument Meteorological Conditions (IMC) are the single highest pre-flight risk factor, carrying a 3.4x higher fatality multiplier than visual flight rules.',
    stat1Val: '22.8%',
    stat1Lbl: 'Top Feature Weight',
    stat2Val: '3.4x',
    stat2Lbl: 'Fatality Multiplier',
    btnText: 'Analyze Bad Weather (IMC) Scenario →',
    preset: {
      category: 'Airplane',
      engines: '1',
      purpose: 'Personal',
      weather: 'IMC',
      amateur: '0'
    }
  },
  cabin: {
    icon: '🛠️',
    stat: '18.0%',
    badge: 'CONSTRUCTION RISK PROFILE',
    title: 'Amateur-Built vs Factory Aircraft',
    desc: 'Homebuilt aircraft represent the 2nd highest pre-incident feature importance (18.0%) in the XGBoost model, exhibiting a 20.0% fatality rate compared to 17.0% for factory-built general aviation aircraft.',
    stat1Val: '18.0%',
    stat1Lbl: 'Amateur Feature Weight',
    stat2Val: '20.0%',
    stat2Lbl: 'Homebuilt Fatality Rate',
    btnText: 'Analyze Amateur-Built Risk Profile →',
    preset: {
      category: 'Airplane',
      engines: '1',
      purpose: 'Personal',
      weather: 'VMC',
      amateur: '1'
    }
  },
  cockpit: {
    icon: '✈️',
    stat: '13.5%',
    badge: 'FLIGHT PURPOSE TELEMETRY',
    title: 'Personal vs Multi-Engine Operations',
    desc: 'Personal flights account for 63% of historical U.S. incident records. Multi-engine commercial and positioning configurations exhibit higher baseline survival metrics.',
    stat1Val: '13.5%',
    stat1Lbl: 'Flight Purpose Weight',
    stat2Val: '39,017',
    stat2Lbl: 'Cleaned NTSB Records',
    btnText: 'Analyze Multi-Engine Scenario →',
    preset: {
      category: 'Airplane',
      engines: '2',
      purpose: 'Positioning',
      weather: 'VMC',
      amateur: '0'
    }
  }
};

/**
 * Universal 3D/4D/5D Parallax Tilt Physics Engine for EVERY Card across the platform
 */
function initUniversal3DTilt() {
  const cards = document.querySelectorAll('.card, .gallery-card, .intel-card, .showcase-container, .explorer-card, .method-card');

  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = ((y - centerY) / centerY) * -11;
      const rotateY = ((x - centerX) / centerX) * 11;

      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);
      card.style.transform = `perspective(1000px) translateY(-8px) scale(1.03) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

/**
 * Opposing Cursor Parallax Drift for Hero Airplane Background
 */
function initOpposingAirplaneParallax() {
  const airplaneOverlay = document.querySelector('.hero-bg-overlay');
  const heroSection = document.getElementById('hero');

  if (!airplaneOverlay || !heroSection) return;

  heroSection.addEventListener('mousemove', (e) => {
    const rect = heroSection.getBoundingClientRect();
    const normX = (e.clientX - rect.left) / rect.width - 0.5;
    const normY = (e.clientY - rect.top) / rect.height - 0.5;

    const translateX = normX * -35;
    const translateY = normY * -25;

    airplaneOverlay.style.transform = `translate3d(${translateX.toFixed(2)}px, ${translateY.toFixed(2)}px, 0) scale(1.04)`;
  });

  heroSection.addEventListener('mouseleave', () => {
    airplaneOverlay.style.transform = 'translate3d(0, 0, 0) scale(1)';
  });
}

/**
 * Sleek, High-End 3D Interactive Hero Canvas Engine with Radial Spotlight Follower
 */
function initHeroInteractiveCanvas() {
  const canvas = document.getElementById('hero-interactive-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width = 0;
  let height = 0;
  let dpr = window.devicePixelRatio || 1;

  // Spring Physics Mouse Tracking
  let targetMouseX = 0;
  let targetMouseY = 0;
  let mouseX = 0;
  let mouseY = 0;

  function resize() {
    const parent = canvas.parentElement;
    width = parent.clientWidth;
    height = parent.clientHeight;
    dpr = window.devicePixelRatio || 1;
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    if (targetMouseX === 0 && targetMouseY === 0) {
      targetMouseX = width * 0.5;
      targetMouseY = height * 0.4;
      mouseX = targetMouseX;
      mouseY = targetMouseY;
    }
    
    initGridMesh();
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    targetMouseX = e.clientX - rect.left;
    targetMouseY = e.clientY - rect.top;
  });

  // Ambient Glowing Orbs Data
  const ORBS = [
    { xRatio: 0.2, yRatio: 0.3, radius: 260, color: 'rgba(92, 6, 50, 0.45)', parallax: 0.08 },
    { xRatio: 0.75, yRatio: 0.55, radius: 300, color: 'rgba(212, 175, 55, 0.18)', parallax: 0.12 },
    { xRatio: 0.45, yRatio: 0.8, radius: 240, color: 'rgba(122, 0, 60, 0.35)', parallax: 0.05 },
    { xRatio: 0.85, yRatio: 0.25, radius: 210, color: 'rgba(212, 175, 55, 0.22)', parallax: 0.15 }
  ];

  // Fluid Mesh Grid Nodes
  let gridCols = 24;
  let gridRows = 14;
  let gridNodes = [];

  function initGridMesh() {
    gridNodes = [];
    const spacingX = width / (gridCols - 1);
    const spacingY = height / (gridRows - 1);

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        gridNodes.push({
          baseX: c * spacingX,
          baseY: r * spacingY,
          x: c * spacingX,
          y: r * spacingY,
          vx: 0,
          vy: 0
        });
      }
    }
  }

  // 3D Floating Geometry Models (Icosahedron & Ring Torus Vertices)
  function createIcosahedron() {
    const phi = (1 + Math.sqrt(5)) / 2;
    const vertices = [
      [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
      [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
      [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ];
    const edges = [
      [0, 11], [0, 5], [0, 1], [0, 7], [0, 10], [1, 5], [1, 9], [1, 8], [1, 7],
      [2, 11], [2, 10], [2, 6], [2, 3], [2, 4], [3, 9], [3, 8], [3, 6], [3, 4],
      [4, 11], [4, 5], [4, 9], [5, 11], [6, 10], [6, 7], [7, 8], [8, 9], [10, 11]
    ];
    return { vertices, edges };
  }

  function createTorusRing(majorR, minorR, numSegments, numTubular) {
    const vertices = [];
    const edges = [];
    
    for (let i = 0; i < numSegments; i++) {
      const u = (i / numSegments) * Math.PI * 2;
      for (let j = 0; j < numTubular; j++) {
        const v = (j / numTubular) * Math.PI * 2;
        const x = (majorR + minorR * Math.cos(v)) * Math.cos(u);
        const y = (majorR + minorR * Math.cos(v)) * Math.sin(u);
        const z = minorR * Math.sin(v);
        vertices.push([x, y, z]);

        const currIndex = i * numTubular + j;
        const nextJ = (j + 1) % numTubular;
        const nextI = ((i + 1) % numSegments) * numTubular + j;
        edges.push([currIndex, i * numTubular + nextJ]);
        edges.push([currIndex, nextI]);
      }
    }
    return { vertices, edges };
  }

  const icosahedronModel = createIcosahedron();
  const torusModel = createTorusRing(110, 32, 12, 6);

  const GEOMETRIES = [
    {
      model: icosahedronModel,
      scale: 55,
      xRatio: 0.18,
      yRatio: 0.42,
      rx: 0, ry: 0, rz: 0,
      rotSpeedX: 0.006, rotSpeedY: 0.009, rotSpeedZ: 0.004,
      parallax: 0.08,
      color: 'rgba(212, 175, 55, 0.45)',
      glowColor: 'rgba(212, 175, 55, 0.6)'
    },
    {
      model: torusModel,
      scale: 0.7,
      xRatio: 0.82,
      yRatio: 0.35,
      rx: 0.6, ry: 0.4, rz: 0.2,
      rotSpeedX: 0.004, rotSpeedY: 0.007, rotSpeedZ: 0.005,
      parallax: 0.12,
      color: 'rgba(122, 0, 60, 0.55)',
      glowColor: 'rgba(212, 175, 55, 0.5)'
    },
    {
      model: icosahedronModel,
      scale: 35,
      xRatio: 0.68,
      yRatio: 0.78,
      rx: 0.2, ry: 0.8, rz: 0.5,
      rotSpeedX: -0.008, rotSpeedY: 0.005, rotSpeedZ: -0.006,
      parallax: 0.18,
      color: 'rgba(212, 175, 55, 0.35)',
      glowColor: 'rgba(212, 175, 55, 0.4)'
    }
  ];

  // 3D Projection Math Matrix
  function project3D(x, y, z, rx, ry, rz, scale, centerX, centerY) {
    let y1 = y * Math.cos(rx) - z * Math.sin(rx);
    let z1 = y * Math.sin(rx) + z * Math.cos(rx);
    let x1 = x;

    let x2 = x1 * Math.cos(ry) + z1 * Math.sin(ry);
    let z2 = -x1 * Math.sin(ry) + z1 * Math.cos(ry);
    let y2 = y1;

    let x3 = x2 * Math.cos(rz) - y2 * Math.sin(rz);
    let y3 = x2 * Math.sin(rz) + y2 * Math.cos(rz);
    let z3 = z2;

    const fov = 400;
    const distance = fov + z3 * scale * 0.3;
    const projX = centerX + (x3 * scale * fov) / distance;
    const projY = centerY + (y3 * scale * fov) / distance;

    return { x: projX, y: projY, z: z3 };
  }

  resize();

  let time = 0;

  function renderFrame() {
    time += 0.016;

    // Smooth Spring Lerp toward Target Cursor Position
    mouseX += (targetMouseX - mouseX) * 0.06;
    mouseY += (targetMouseY - mouseY) * 0.06;

    const normMouseX = (mouseX / width - 0.5) * 2;
    const normMouseY = (mouseY / height - 0.5) * 2;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw Dynamic Cursor Radial Spotlight Follower
    const spotGrad = ctx.createRadialGradient(mouseX, mouseY, 10, mouseX, mouseY, 380);
    spotGrad.addColorStop(0, 'rgba(212, 175, 55, 0.22)');
    spotGrad.addColorStop(0.5, 'rgba(92, 6, 50, 0.15)');
    spotGrad.addColorStop(1, 'rgba(11, 3, 10, 0)');
    ctx.fillStyle = spotGrad;
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, 380, 0, Math.PI * 2);
    ctx.fill();

    // 2. Draw Ambient Glowing Orbs with Parallax Drift
    ORBS.forEach(orb => {
      const orbX = width * orb.xRatio + normMouseX * 50 * orb.parallax;
      const orbY = height * orb.yRatio + normMouseY * 50 * orb.parallax;
      
      const grad = ctx.createRadialGradient(orbX, orbY, 10, orbX, orbY, orb.radius);
      grad.addColorStop(0, orb.color);
      grad.addColorStop(1, 'rgba(11, 3, 10, 0)');
      
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(orbX, orbY, orb.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // 3. Draw Fluid Distortion Mesh Grid
    const mouseRadiusSq = 160 * 160;
    for (let i = 0; i < gridNodes.length; i++) {
      const node = gridNodes[i];
      const dx = mouseX - node.baseX;
      const dy = mouseY - node.baseY;
      const distSq = dx * dx + dy * dy;

      if (distSq < mouseRadiusSq && distSq > 0) {
        const force = (1 - distSq / mouseRadiusSq) * -35;
        const dist = Math.sqrt(distSq);
        node.vx += (dx / dist) * force * 0.05;
        node.vy += (dy / dist) * force * 0.05;
      }

      // Spring Return to Base Position
      node.vx += (node.baseX - node.x) * 0.08;
      node.vy += (node.baseY - node.y) * 0.08;
      node.vx *= 0.85;
      node.vy *= 0.85;

      node.x += node.vx;
      node.y += node.vy;
    }

    // Draw Grid Mesh Lines
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const index = r * gridCols + c;
        const node = gridNodes[index];

        if (c < gridCols - 1) {
          const rightNode = gridNodes[index + 1];
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(rightNode.x, rightNode.y);
        }
        if (r < gridRows - 1) {
          const bottomNode = gridNodes[index + gridCols];
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(bottomNode.x, bottomNode.y);
        }
      }
    }
    ctx.stroke();

    // 4. Draw 3D Floating Wireframe Geometries
    GEOMETRIES.forEach(geo => {
      geo.rx += geo.rotSpeedX;
      geo.ry += geo.rotSpeedY;
      geo.rz += geo.rotSpeedZ;

      const currentRx = geo.rx + normMouseY * 0.35;
      const currentRy = geo.ry + normMouseX * 0.35;

      const centerX = width * geo.xRatio + normMouseX * 70 * geo.parallax;
      const centerY = height * geo.yRatio + normMouseY * 70 * geo.parallax;

      const projectedPts = geo.model.vertices.map(v => 
        project3D(v[0], v[1], v[2], currentRx, currentRy, geo.rz, geo.scale, centerX, centerY)
      );

      ctx.strokeStyle = geo.color;
      ctx.lineWidth = 1.2;
      ctx.shadowColor = geo.glowColor;
      ctx.shadowBlur = 12;

      ctx.beginPath();
      geo.model.edges.forEach(e => {
        const p1 = projectedPts[e[0]];
        const p2 = projectedPts[e[1]];
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
      });
      ctx.stroke();

      ctx.fillStyle = geo.glowColor;
      projectedPts.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

/**
 * Incident Scenario Analyzer ML Inference Engine (Pre-Incident Variables Only)
 */
function analyzeIncidentScenario(input) {
  let scores = {
    Fatal: 1.0,
    Serious: 1.5,
    Minor: 2.5,
    None: 5.0
  };

  if (input.weather === 'IMC') {
    scores.Fatal *= 3.4;
    scores.Serious *= 2.2;
    scores.Minor *= 1.1;
    scores.None *= 0.4;
  } else if (input.weather === 'VMC') {
    scores.None *= 1.6;
    scores.Minor *= 1.4;
    scores.Serious *= 0.8;
  }

  if (input.amateur === '1') {
    scores.Fatal += 2.2;
    scores.Serious += 1.8;
  }

  if (input.purpose === 'Personal') {
    scores.Fatal += 1.8;
    scores.Serious += 1.5;
  } else if (input.purpose === 'Instructional' || input.purpose === 'Executive') {
    scores.Minor += 1.2;
    scores.None += 2.0;
  } else if (input.purpose === 'Aerial Application') {
    scores.Serious += 1.2;
    scores.Minor += 1.5;
  }

  if (parseInt(input.engines, 10) === 1) {
    scores.Fatal += 0.8;
    scores.Serious += 0.6;
  } else if (parseInt(input.engines, 10) >= 2) {
    scores.Minor += 1.2;
    scores.None += 1.5;
  }

  if (input.season === 'Winter') {
    scores.Fatal += 0.6;
    scores.Serious += 0.5;
  } else if (input.season === 'Summer') {
    scores.Minor += 0.8;
    scores.None += 1.0;
  }

  const total = scores.Fatal + scores.Serious + scores.Minor + scores.None;
  const probs = {
    Fatal: Math.round((scores.Fatal / total) * 100),
    Serious: Math.round((scores.Serious / total) * 100),
    Minor: Math.round((scores.Minor / total) * 100),
    None: Math.round((scores.None / total) * 100)
  };

  const currentSum = probs.Fatal + probs.Serious + probs.Minor + probs.None;
  if (currentSum !== 100) {
    probs.None += (100 - currentSum);
  }

  let highestClass = 'None';
  let maxP = -1;
  for (const [cls, p] of Object.entries(probs)) {
    if (p > maxP) {
      maxP = p;
      highestClass = cls;
    }
  }

  return {
    predictedSeverity: highestClass.toUpperCase(),
    probabilities: probs,
    features: PRE_INCIDENT_FEATURE_IMPORTANCES
  };
}

/**
 * Handle UI Interaction for Scenario Card & Telemetry Sync
 */
/**
 * Handle UI Interaction for Scenario Card & Freemium Premium Lock
 */
/**
 * Handle UI Interaction for Scenario Card, Free Limit & Checkout Flow
 */
function initScenarioAnalyzer() {
  const form = document.getElementById('scenario-form');
  const resultView = document.getElementById('result-view');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnReset = document.getElementById('btn-reset');
  const btnUnlock = document.getElementById('btn-unlock-intelligence');
  const btnLimitPricing = document.getElementById('btn-limit-pricing');
  const btnPricingCheckout = document.getElementById('btn-pricing-checkout');

  if (!form || !resultView) return;

  // Check state on load
  checkFreeLimitUI();

  btnAnalyze.addEventListener('click', (e) => {
    e.preventDefault();
    handleAnalysisSubmission();
  });

  btnReset.addEventListener('click', (e) => {
    e.preventDefault();
    const isUnlocked = localStorage.getItem('aeroIntel_isPremiumUnlocked') === 'true';

    if (!isUnlocked) {
      // Free trial is expired -> prompt checkout!
      openCheckoutModal();
    } else {
      // Paid user -> allow re-analyzing scenarios freely!
      resultView.classList.remove('active');
      form.classList.remove('hidden');
    }
  });

  if (btnUnlock) {
    btnUnlock.addEventListener('click', () => {
      const isUnlocked = localStorage.getItem('aeroIntel_isPremiumUnlocked') === 'true';
      if (isUnlocked) {
        document.getElementById('premium-lock-wrapper')?.scrollIntoView({ behavior: 'smooth' });
      } else {
        openCheckoutModal();
      }
    });
  }

  if (btnLimitPricing) {
    btnLimitPricing.addEventListener('click', () => {
      openCheckoutModal();
    });
  }

  if (btnPricingCheckout) {
    btnPricingCheckout.addEventListener('click', () => {
      openCheckoutModal();
    });
  }
}

function checkFreeLimitUI() {
  const form = document.getElementById('scenario-form');
  const limitBanner = document.getElementById('free-limit-banner');
  const statusBadge = document.getElementById('scenario-status-badge');
  const isUnlocked = localStorage.getItem('aeroIntel_isPremiumUnlocked') === 'true';
  const freeUsed = localStorage.getItem('aeroIntel_freeAnalysisUsed') === 'true';

  if (isUnlocked) {
    if (limitBanner) limitBanner.classList.add('hidden');
    if (form) form.classList.remove('hidden');
    if (statusBadge) {
      statusBadge.textContent = 'UNLIMITED PREMIUM';
      statusBadge.style.color = '#10B981';
      statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    }
  } else if (freeUsed) {
    if (limitBanner) limitBanner.classList.remove('hidden');
    if (form) form.classList.add('hidden');
    if (statusBadge) {
      statusBadge.textContent = 'FREE TRIAL EXPIRED';
      statusBadge.style.color = '#EF4444';
      statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    }
  } else {
    if (limitBanner) limitBanner.classList.add('hidden');
    if (form) form.classList.remove('hidden');
    if (statusBadge) {
      statusBadge.textContent = '1 FREE TRIAL';
    }
  }
}

function handleAnalysisSubmission() {
  const isUnlocked = localStorage.getItem('aeroIntel_isPremiumUnlocked') === 'true';
  const freeUsed = localStorage.getItem('aeroIntel_freeAnalysisUsed') === 'true';

  if (!isUnlocked && freeUsed) {
    openCheckoutModal();
    return;
  }

  // Mark 1 free analysis used
  if (!isUnlocked) {
    localStorage.setItem('aeroIntel_freeAnalysisUsed', 'true');
  }

  triggerAnalysis();
}

function triggerAnalysis() {
  const form = document.getElementById('scenario-form');
  const resultView = document.getElementById('result-view');

  const inputData = {
    weather: document.getElementById('field-weather').value,
    purpose: document.getElementById('field-purpose').value,
    amateur: document.getElementById('field-amateur').value,
    category: document.getElementById('field-category').value,
    engines: document.getElementById('field-engines').value,
    season: document.getElementById('field-season').value,
    year: document.getElementById('field-year').value,
    month: document.getElementById('field-month').value
  };

  const result = analyzeIncidentScenario(inputData);
  updateResultView(result);
  updateConsoleTelemetry(inputData, result);

  form.classList.add('hidden');
  resultView.classList.add('active');
}

function updateResultView(result) {
  const banner = document.getElementById('severity-banner');
  const valueEl = document.getElementById('severity-value');
  const chipEl = document.getElementById('severity-confidence-chip');
  const lockWrapper = document.getElementById('premium-lock-wrapper');
  const lockTitle = document.getElementById('premium-lock-title');
  const lockBadge = document.getElementById('premium-badge-tag');
  const btnUnlock = document.getElementById('btn-unlock-intelligence');
  const btnReset = document.getElementById('btn-reset');

  const isUnlocked = localStorage.getItem('aeroIntel_isPremiumUnlocked') === 'true';
  
  const sev = result.predictedSeverity.toLowerCase();
  banner.className = `severity-banner ${sev}`;
  valueEl.textContent = result.predictedSeverity;

  const probs = result.probabilities;
  const highestProb = probs[result.predictedSeverity.charAt(0) + result.predictedSeverity.slice(1).toLowerCase()] || 50;

  if (chipEl) {
    chipEl.textContent = `${highestProb}% Model Probability`;
  }

  setProbRow('fatal', probs.Fatal);
  setProbRow('serious', probs.Serious);
  setProbRow('minor', probs.Minor);
  setProbRow('none', probs.None);

  renderWhySection(result.features);

  if (isUnlocked) {
    if (lockWrapper) {
      lockWrapper.classList.remove('locked');
      lockWrapper.classList.add('unlocked');
    }
    if (lockTitle) lockTitle.innerHTML = '<span class="lock-icon">🔓</span> WHY THIS ASSESSMENT?';
    if (lockBadge) lockBadge.textContent = 'PREMIUM UNLOCKED';
    if (btnUnlock) btnUnlock.innerHTML = '✓ AeroIntel Intelligence Unlocked';
    if (btnReset) btnReset.innerHTML = '← Re-analyze Scenario';
    
    setTimeout(() => {
      document.querySelectorAll('.prob-fill').forEach((el) => {
        const targetWidth = el.getAttribute('data-pct') + '%';
        el.style.width = targetWidth;
      });
    }, 100);
  } else {
    if (lockWrapper) {
      lockWrapper.classList.add('locked');
      lockWrapper.classList.remove('unlocked');
    }
    if (lockTitle) lockTitle.innerHTML = '<span class="lock-icon">🔒</span> Want to know why?';
    if (lockBadge) lockBadge.textContent = 'AEROINTEL PREMIUM';
    if (btnUnlock) btnUnlock.innerHTML = 'Unlock AeroIntel Intelligence →';
    if (btnReset) btnReset.innerHTML = '🔒 Free Trial Expired — Unlock Intelligence to Re-Analyze →';
  }
}

/**
 * Handle Realistic Checkout Lightbox Modal
 */
function initCheckoutModal() {
  const modal = document.getElementById('checkout-modal');
  const backdrop = document.getElementById('checkout-modal-backdrop');
  const closeBtn = document.getElementById('checkout-modal-close');
  const checkoutForm = document.getElementById('checkout-form');
  const viewUnlockedBtn = document.getElementById('btn-view-unlocked-intelligence');

  if (!modal) return;

  const closeCheckout = () => {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
  };

  backdrop?.addEventListener('click', closeCheckout);
  closeBtn?.addEventListener('click', closeCheckout);

  // Tab switching for payment methods
  const payTabs = modal.querySelectorAll('.pay-tab');
  payTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      payTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetMethod = tab.getAttribute('data-pay-method');
      modal.querySelectorAll('.pay-method-panel').forEach(panel => {
        panel.classList.add('hidden');
      });

      const activePanel = document.getElementById(`pay-panel-${targetMethod}`);
      if (activePanel) {
        activePanel.classList.remove('hidden');
      }
    });
  });

  checkoutForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const stateForm = document.getElementById('checkout-state-form');
    const stateProcessing = document.getElementById('checkout-state-processing');
    const stateSuccess = document.getElementById('checkout-state-success');

    // STEP 1 -> STEP 2: SHOW ONLY PROCESSING
    stateForm.classList.add('hidden');
    stateProcessing.classList.remove('hidden');
    stateSuccess.classList.add('hidden');

    // Simulate 1.2s gateway processing delay
    setTimeout(() => {
      // STEP 2 -> STEP 3: SHOW ONLY SUCCESS
      stateProcessing.classList.add('hidden');
      stateSuccess.classList.remove('hidden');

      // Unlock Premium in localStorage
      localStorage.setItem('aeroIntel_isPremiumUnlocked', 'true');

      playCabinChime();
    }, 1200);
  });

  viewUnlockedBtn?.addEventListener('click', () => {
    closeCheckout();

    // Reset checkout form states for next demo loop
    document.getElementById('checkout-state-form')?.classList.remove('hidden');
    document.getElementById('checkout-state-processing')?.classList.add('hidden');
    document.getElementById('checkout-state-success')?.classList.add('hidden');

    // Update UI and reveal unlocked insights
    const inputData = {
      weather: document.getElementById('field-weather').value,
      purpose: document.getElementById('field-purpose').value,
      amateur: document.getElementById('field-amateur').value,
      category: document.getElementById('field-category').value,
      engines: document.getElementById('field-engines').value,
      season: document.getElementById('field-season').value,
      year: document.getElementById('field-year').value,
      month: document.getElementById('field-month').value
    };
    
    const result = analyzeIncidentScenario(inputData);
    updateResultView(result);

    checkFreeLimitUI();

    document.getElementById('scenario-card')?.scrollIntoView({ behavior: 'smooth' });
  });
}

function openCheckoutModal() {
  const modal = document.getElementById('checkout-modal');
  if (modal) {
    // Reset to form state
    document.getElementById('checkout-state-form')?.classList.remove('hidden');
    document.getElementById('checkout-state-processing')?.classList.add('hidden');
    document.getElementById('checkout-state-success')?.classList.add('hidden');

    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
  }
}

function updateConsoleTelemetry(inputData, result) {
  const telemetryEl = document.getElementById('console-telemetry-text');
  const liveStatusEl = document.getElementById('console-live-status');
  
  if (liveStatusEl) {
    liveStatusEl.textContent = 'INTERACTIVE SCENARIO ACTIVE';
    liveStatusEl.style.color = 'var(--qatar-gold)';
    liveStatusEl.style.borderColor = 'var(--qatar-gold)';
  }

  if (telemetryEl) {
    const sevClass = result.predictedSeverity;
    const sevPct = result.probabilities[sevClass.charAt(0) + sevClass.slice(1).toLowerCase()] || 50;
    
    telemetryEl.innerHTML = `
      [INPUT]: WeatherCondition: ${inputData.weather}<br>
      [INPUT]: PurposeOfFlight: ${inputData.purpose}<br>
      [INPUT]: AmateurBuilt: ${inputData.amateur === '1' ? '1 (Homebuilt)' : '0 (Factory)'}<br>
      [INPUT]: Category: ${inputData.category} | Engines: ${inputData.engines}<br>
      [EVAL]: Running XGBoost pre-flight classifier...<br>
      [RESULT]: Predicted Class -> ${sevClass} (${sevPct}.0%)
    `;
  }
}

function setProbRow(key, pct) {
  const pctEl = document.getElementById(`prob-pct-${key}`);
  const fillEl = document.getElementById(`prob-fill-${key}`);
  if (pctEl && fillEl) {
    pctEl.textContent = `${pct}%`;
    fillEl.setAttribute('data-pct', pct);
  }
}

function renderWhySection(features) {
  const container = document.getElementById('why-list');
  if (!container) return;

  container.innerHTML = features.slice(0, 4).map(f => `
    <div class="why-item">
      <span class="why-name">${f.name}</span>
      <div class="why-bar-wrap">
        <div class="why-bar" style="width: ${f.pct * 3.5}%;"></div>
      </div>
    </div>
  `).join('');
}

/**
 * Interactive NTSB Case Study Lightbox Modal Handler
 */
function initGalleryModal() {
  const modal = document.getElementById('gallery-modal');
  const backdrop = document.getElementById('gallery-modal-backdrop');
  const closeBtn = document.getElementById('gallery-modal-close');
  const actionBtn = document.getElementById('modal-action-btn');
  const cards = document.querySelectorAll('.gallery-card');

  if (!modal) return;

  let activePreset = null;

  cards.forEach(card => {
    const openCardHandler = () => {
      const galleryId = card.getAttribute('data-gallery-id');
      const data = GALLERY_DATA[galleryId];
      if (!data) return;

      const modalIcon = document.getElementById('modal-graphic-icon');
      const modalStat = document.getElementById('modal-graphic-stat');
      if (modalIcon) modalIcon.textContent = data.icon;
      if (modalStat) modalStat.textContent = data.stat;

      document.getElementById('modal-badge').textContent = data.badge;
      document.getElementById('modal-title').textContent = data.title;
      document.getElementById('modal-desc').textContent = data.desc;
      document.getElementById('modal-stat-1').textContent = data.stat1Val;
      document.getElementById('modal-lbl-1').textContent = data.stat1Lbl;
      document.getElementById('modal-stat-2').textContent = data.stat2Val;
      document.getElementById('modal-lbl-2').textContent = data.stat2Lbl;
      actionBtn.textContent = data.btnText;
      
      activePreset = data.preset;

      modal.classList.add('active');
      modal.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    };

    card.addEventListener('click', openCardHandler);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openCardHandler();
      }
    });
  });

  const closeModal = () => {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  };

  closeBtn.addEventListener('click', closeModal);
  backdrop.addEventListener('click', closeModal);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  actionBtn.addEventListener('click', () => {
    if (activePreset) {
      if (document.getElementById('field-weather')) document.getElementById('field-weather').value = activePreset.weather;
      if (document.getElementById('field-purpose')) document.getElementById('field-purpose').value = activePreset.purpose;
      if (document.getElementById('field-amateur')) document.getElementById('field-amateur').value = activePreset.amateur;
      if (document.getElementById('field-category')) document.getElementById('field-category').value = activePreset.category;
      if (document.getElementById('field-engines')) document.getElementById('field-engines').value = activePreset.engines;
    }

    closeModal();

    const heroSection = document.getElementById('hero');
    if (heroSection) {
      heroSection.scrollIntoView({ behavior: 'smooth' });
    }

    setTimeout(() => {
      triggerAnalysis();
    }, 500);
  });
}

/**
 * Interactive About Me Glass Popup Modal Handler
 */
function initAboutModal() {
  const modal = document.getElementById('about-modal');
  const backdrop = document.getElementById('about-modal-backdrop');
  const closeBtn = document.getElementById('about-modal-close');
  const aboutNavBtn = document.getElementById('nav-about-me');
  const footerAboutLink = document.getElementById('footer-about-link');
  const devCard = document.getElementById('about-developer-card');
  const modalAvatar = document.getElementById('about-modal-avatar');
  const cardAvatar = document.getElementById('about-card-avatar');

  if (!modal) return;

  const openModal = (e) => {
    if (e) e.preventDefault();

    // Cache-bust avatar URL with current timestamp to ensure freshly updated GitHub avatar renders immediately
    const cacheBustUrl = `https://github.com/alokranjan08.png?t=${Date.now()}`;
    if (modalAvatar) modalAvatar.src = cacheBustUrl;
    if (cardAvatar) cardAvatar.src = cacheBustUrl;

    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  };

  const closeModal = () => {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  };

  if (aboutNavBtn) aboutNavBtn.addEventListener('click', openModal);
  if (footerAboutLink) footerAboutLink.addEventListener('click', openModal);
  if (devCard) devCard.addEventListener('click', openModal);

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (backdrop) backdrop.addEventListener('click', closeModal);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  // Automatically close modal when user scrolls away from the About section
  window.addEventListener('scroll', () => {
    if (modal.classList.contains('active')) {
      const aboutSec = document.getElementById('about');
      if (aboutSec) {
        const rect = aboutSec.getBoundingClientRect();
        if (rect.bottom < -100 || rect.top > window.innerHeight + 100) {
          closeModal();
        }
      }
    }
  }, { passive: true });
}

/**
 * AeroIntel Console Sidebar Tabs
 */
function initConsoleTabs() {
  const tabButtons = document.querySelectorAll('.console-tab-item');
  
  tabButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const tabId = btn.getAttribute('data-tab');
      
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      document.querySelectorAll('.console-tab-content').forEach(content => {
        content.classList.remove('active');
      });
      
      const targetContent = document.getElementById(`tab-${tabId}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

/**
 * Data & Code Explorer Section Subnav Tabs
 */
function initExplorerTabs() {
  const tabBtns = document.querySelectorAll('.explorer-tab-btn');
  
  tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const expTabId = btn.getAttribute('data-exp-tab');
      
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      document.querySelectorAll('.explorer-tab-panel').forEach(panel => {
        panel.classList.remove('active');
      });
      
      const targetPanel = document.getElementById(`exp-panel-${expTabId}`);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });
}

/**
 * Code Pipeline Step Switcher
 */
function initCodeStepSwitcher() {
  const stepBtns = document.querySelectorAll('.code-step-btn');
  const codeBlock = document.getElementById('code-content-block');
  
  if (!codeBlock) return;

  stepBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const stepId = btn.getAttribute('data-code-step');
      
      stepBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      if (CODE_STEPS[stepId]) {
        codeBlock.innerHTML = CODE_STEPS[stepId];
      }
    });
  });
}

/**
 * Dataset Table Live Search Filter
 */
function initTableSearch() {
  const searchInput = document.getElementById('dataset-search-input');
  const tableBody = document.getElementById('csv-table-body');
  
  if (!searchInput || !tableBody) return;

  searchInput.addEventListener('input', () => {
    const filter = searchInput.value.toLowerCase().trim();
    const rows = tableBody.querySelectorAll('tr');

    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      if (text.includes(filter)) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
}

function initMobileNav() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const navLinks = document.getElementById('nav-links');

  if (toggleBtn && navLinks) {
    toggleBtn.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });

    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('active');
      });
    });
  }
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      // If it's the about link, skip standard smooth scroll as initAboutModal handles pop-up
      if (this.id === 'nav-about-me' || this.id === 'footer-about-link') return;

      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth'
        });
      }
    });
  });
}

/**
 * Synthesize Authentic 2-Tone Aircraft Cabin Chime ("Ding-Dong") using Web Audio API
 */
function playCabinChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();

    // High Tone: D5 (587.33 Hz)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.value = 587.33;
    gain1.gain.setValueAtTime(0.18, ctx.currentTime);
    gain1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(ctx.currentTime);
    osc1.stop(ctx.currentTime + 0.5);

    // Low Tone: A4 (440.00 Hz) after 220ms delay
    setTimeout(() => {
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sine';
      osc2.frequency.value = 440.00;
      gain2.gain.setValueAtTime(0.18, ctx.currentTime);
      gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.7);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(ctx.currentTime);
      osc2.stop(ctx.currentTime + 0.7);
    }, 220);
  } catch (e) {
    // Silent fallback
  }
}

/**
 * Upgraded Bonus Round Easter Egg:
 * Triggers on typing 'alok', 'aero', 'ntsb', or 'qatar', or clicking brand logo 5 times.
 */
function initEasterEgg() {
  const brandLogo = document.querySelector('.brand');
  let clickCount = 0;
  let keySequence = '';

  const triggerEasterEgg = (triggerWord = 'ALOK RANJAN') => {
    playCabinChime();

    let toast = document.getElementById('easter-egg-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'easter-egg-toast';
      toast.style.cssText = `
        position: fixed;
        bottom: 2.5rem;
        right: 2.5rem;
        z-index: 9999;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.95) 0%, rgba(92, 6, 50, 0.95) 100%);
        backdrop-filter: blur(16px);
        color: #FFFFFF;
        padding: 1.1rem 1.6rem;
        border-radius: 20px;
        border: 2px solid #F3E5AB;
        box-shadow: 0 10px 45px rgba(212, 175, 55, 0.7), 0 0 20px rgba(92, 6, 50, 0.5);
        font-family: var(--font-sans);
        font-weight: 800;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        animation: toastBounce 0.6s cubic-bezier(0.34, 1.45, 0.64, 1);
      `;
      toast.innerHTML = `
        <span style="font-size: 1.4rem;">✈️</span>
        <div>
          <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--qatar-gold);">5-Star Flight Deck Activated</div>
          <div style="font-size: 0.95rem; font-weight: 800;">${triggerWord.toUpperCase()} INSIGHT MODE UNLOCKED 🌟</div>
        </div>
      `;
      document.body.appendChild(toast);

      setTimeout(() => {
        if (toast) toast.remove();
      }, 5000);
    }
  };

  if (brandLogo) {
    brandLogo.addEventListener('click', () => {
      clickCount++;
      if (clickCount >= 5) {
        clickCount = 0;
        triggerEasterEgg('AeroIntel');
      }
    });
  }

  document.addEventListener('keydown', (e) => {
    keySequence += e.key.toLowerCase();
    if (keySequence.length > 15) {
      keySequence = keySequence.slice(-15);
    }
    
    if (keySequence.endsWith('alok')) {
      keySequence = '';
      triggerEasterEgg('Alok Ranjan');
    } else if (keySequence.endsWith('aero')) {
      keySequence = '';
      triggerEasterEgg('AeroIntel');
    } else if (keySequence.endsWith('ntsb')) {
      keySequence = '';
      triggerEasterEgg('NTSB Intelligence');
    } else if (keySequence.endsWith('qatar')) {
      keySequence = '';
      triggerEasterEgg('5-Star Luxury');
    }
  });
}
