import type { DesignSystem, Page, SlideMeta, SlideTransition } from '@open-slide/core';
import type { CSSProperties, ReactNode } from 'react';
import consulta from './assets/consulta-juridica.jpg';

export const meta: SlideMeta = {
  title: 'Lawten · Sustentación final',
  createdAt: '2026-11-19T23:19:32.762Z',
};

export const design: DesignSystem = {
  palette: { bg: '#F3F0E5', text: '#142A20', accent: '#17644C' },
  fonts: { display: 'Georgia, serif', body: 'Segoe UI, Arial, sans-serif' },
  typeScale: { hero: 178, body: 37 },
  radius: 8,
};

export const transition: SlideTransition = {
  duration: 240,
  easing: 'cubic-bezier(0, 0, 0.2, 1)',
  enter: { keyframes: [{ opacity: 0 }, { opacity: 1 }] },
};

const paper = 'var(--osd-bg, #F3F0E5)';
const ink = 'var(--osd-text, #142A20)';
const moss = 'var(--osd-accent, #17644C)';
const dark = '#10271F';
const mint = '#D7EDAA';
const muted = '#40584C';
const paperLine = '#B6C4B5';
const darkLine = '#547161';
const font = 'var(--osd-font-body, Segoe UI, Arial, sans-serif)';
const serif = 'var(--osd-font-display, Georgia, serif)';

const heading: CSSProperties = { fontFamily: serif, fontSize: 86, fontWeight: 400, letterSpacing: -3, lineHeight: 1.04, margin: 0 };
const body: CSSProperties = { fontSize: 'var(--osd-size-body, 37px)', lineHeight: 1.23, margin: 0 };
const label: CSSProperties = { fontSize: 24, lineHeight: 1.2, letterSpacing: 2.8, fontWeight: 660, textTransform: 'uppercase' };

function Canvas({ children, night = false }: { children: ReactNode; night?: boolean }) {
  return <section data-lawten-slide style={{
    width: '100%', height: '100%', position: 'relative', overflow: 'hidden', boxSizing: 'border-box',
    background: night ? dark : paper, color: night ? '#FAF7E9' : ink, fontFamily: font,
    padding: '104px 120px 110px',
  }}>{children}</section>;
}

function Footer({ page, night = false }: { page: string; night?: boolean }) {
  return <footer style={{
    position: 'absolute', left: 120, right: 120, bottom: 56, borderTop: '2px solid ' + (night ? darkLine : paperLine),
    paddingTop: 23, display: 'flex', justifyContent: 'space-between', fontSize: 23, color: night ? '#C3D4C6' : muted,
  }}><span>Lawten</span><span>{page} / 07</span></footer>;
}

const Cover: Page = () => <Canvas night>
  <img src={consulta} alt="Ilustración de una persona que contrasta un libro jurídico con un escrito" style={{
    display: 'block', position: 'absolute', top: 0, right: 0, width: 875, height: 1080, objectFit: 'cover', objectPosition: '57% 52%',
  }} />
  <div style={{ position: 'absolute', top: 0, left: 0, width: 1045, height: 1080, padding: '130px 120px' }}>
    <div style={{ ...label, color: mint }}>Proyecto integrador</div>
    <h1 style={{ ...heading, fontSize: 'var(--osd-size-hero, 178px)', letterSpacing: -10, marginTop: 194 }}>Lawten</h1>
    <p style={{ ...body, fontSize: 47, maxWidth: 760, marginTop: 46 }}>Asistente de consulta de derecho laboral individual</p>
    <div style={{ position: 'absolute', left: 120, right: 104, bottom: 79, borderTop: '2px solid '+darkLine, paddingTop: 23, fontSize: 25, color: '#C3D4C6' }}>Equipo Lawten</div>
  </div>
  <small style={{ position: 'absolute', bottom: 30, right: 34, color: '#F8F4E7', fontSize: 18, background: dark, padding: '9px 14px' }}>
    Ilustración generada con IA
  </small>
</Canvas>;

const Problem: Page = () => <Canvas>
  <h2 style={{ ...heading, maxWidth: 1080 }}>Qué norma revisar<br/>antes de citarla</h2>
  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.02fr', gap: 115, marginTop: 92 }}>
    <div>
      <div style={{ ...label, color: moss }}>La tarea del abogado</div>
      <p style={{ ...body, marginTop: 26, maxWidth: 700 }}>Un abogado junior redacta un escrito. Debe encontrar el artículo pertinente y comprobar que dice lo que piensa citar.</p>
    </div>
    <div style={{ borderLeft: '2px solid '+paperLine, paddingLeft: 74 }}>
      <div style={{ ...label, color: moss }}>Lo que entrega Lawten</div>
      <p style={{ ...body, marginTop: 26, maxWidth: 700 }}>Referencias con enlaces a las fuentes y un resumen de los artículos recuperados.</p>
    </div>
  </div>
  <div style={{ marginTop: 84, width: 1530, borderTop: '2px solid '+paperLine, paddingTop: 33, fontSize: 33, fontWeight: 620 }}>
    Medimos si la norma esperada aparece entre las tres primeras referencias.
  </div>
  <Footer page="02"/>
</Canvas>;

const flow = [
  ['Consulta', 'El abogado describe su situación por escrito.'],
  ['Decisión', 'El sistema elige si usa el buscador.'],
  ['Búsqueda', 'Combina palabras y significado para ordenar artículos de fuentes oficiales.'],
  ['Respuesta', 'Muestra tres referencias y resume lo que dicen.'],
];

const System: Page = () => <Canvas>
  <h2 style={heading}>Cómo funciona Lawten</h2>
  <ol style={{
    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 40, margin: '134px 0 0', padding: 0, listStyle: 'none',
  }}>
    {flow.map(([title, copy], i) => <li key={title} style={{ position: 'relative', borderTop: '3px solid '+moss, paddingTop: 21 }}>
      <span style={{ color: moss, fontWeight: 650, fontSize: 27 }}>0{i+1}</span>
      {i < flow.length - 1 && <span aria-hidden style={{ lineHeight: 1, color: moss, fontSize: 42, position: 'absolute', top: 12, right: 3 }}>→</span>}
      <h3 style={{ ...heading, fontSize: 54, letterSpacing: -1, marginTop: 31 }}>{title}</h3>
      <p style={{ ...body, fontSize: 33, marginTop: 24, paddingRight: 16 }}>{copy}</p>
    </li>)}
  </ol>
  <div style={{ marginTop: 84, marginLeft: 430, padding: '20px 26px', width: 1140, background: '#DCE7D9', fontSize: 33 }}>
    Si decide omitir la búsqueda, devuelve un aviso sin referencias.
  </div>
  <Footer page="03"/>
</Canvas>;

const Demo: Page = () => <Canvas night>
  <h2 style={{ ...heading, fontSize: 112 }}>Demo en vivo</h2>
  <div style={{ marginTop: 110, width: 1580 }}>
    {['Caso fácil', 'Caso difícil', 'Caso adversarial'].map((name) => <div key={name} style={{
      borderTop: '2px solid '+darkLine, display: 'grid', gridTemplateColumns: '535px 1fr', alignItems: 'center', minHeight: 136,
    }}>
      <span style={{ fontFamily: serif, fontSize: 48 }}>{name}</span><span style={{ ...body, color: mint }}>—</span>
    </div>)}
  </div>
  <div style={{ fontSize: 29, color: mint, marginTop: 59 }}>Demo pública &nbsp; —</div>
  <Footer page="04" night/>
</Canvas>;

const toolComparison = [
  { name: 'Búsqueda directa', seconds: 6.2, time: '6,2 s', color: '#A6BAAA' },
  { name: 'Búsqueda con decisión previa', seconds: 8.8, time: '8,8 s', color: '#207154' },
];

const Tool: Page = () => <Canvas>
  <h2 style={heading}>Decidir si buscar tomó más tiempo</h2>
  <p style={{ ...body, color: muted, marginTop: 32 }}>Medianas de la misma sesión, sobre las mismas consultas.</p>
  <div style={{ display: 'grid', gridTemplateColumns: '1040px 1fr', gap: 122, marginTop: 100 }}>
    <div>
      {toolComparison.map(({name, seconds, time, color}) => <div key={name} style={{ marginBottom: 50 }}>
        <h3 style={{ margin: '0 0 17px', fontSize: 31, fontWeight: 600 }}>{name}</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 23 }}>
          <div style={{ height: 58, width: seconds * 94, backgroundColor: color }} aria-hidden />
          <span style={{ fontSize: 35, fontWeight: 650 }}>{time}</span>
        </div>
      </div>)}
    </div>
    <aside style={{ borderLeft: '2px solid '+paperLine, paddingLeft: 60 }}>
      <div style={{ fontFamily: serif, fontSize: 104, lineHeight: 1.03, letterSpacing: -5, color: moss }}>2 de 13</div>
      <p style={{ ...body, marginTop: 23, maxWidth: 440 }}>aciertos en ambas versiones.</p>
    </aside>
  </div>
  <p style={{ fontSize: 33, margin: '35px 0 0', fontWeight: 600 }}>El sistema omitió la búsqueda en dos consultas laborales.</p>
  <Footer page="05"/>
</Canvas>;

const efficiency = [
  ['Pesos originales', '3 de 13', '9,55 s', '12,34 s', '3.087 MB'],
  ['Reducido a 8 bits', '1 de 13', '29,08 s', '34,86 s', '1.777 MB'],
  ['Reducido a 4 bits', '1 de 13', '9,84 s', '14,32 s', '1.122 MB'],
];

const Efficiency: Page = () => <Canvas>
  <h2 style={{ ...heading, fontSize: 79 }}>Un modelo más pequeño, respuestas más lentas</h2>
  <p style={{ ...body, color: muted, marginTop: 32 }}>Pruebas en la misma T4 de Colab. La meta era bajar de 5 s de mediana.</p>
  <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 70, fontSize: 34, textAlign: 'left', fontVariantNumeric: 'tabular-nums' }}>
    <thead>
      <tr>{['Versión', 'Aciertos', 'Mediana', 'Percentil 95', 'Tamaño del modelo'].map((item) => <th key={item} scope="col" style={{
        fontSize: 26, fontWeight: 650, padding: '20px 22px 23px', borderBottom: '2px solid '+paperLine, color: muted,
      }}>{item}</th>)}</tr>
    </thead>
    <tbody>{efficiency.map((row, i) => <tr key={row[0]} style={{ background: i === 0 ? '#DCE7D9' : 'transparent' }}>
      {row.map((value, column) => column === 0
        ? <th scope="row" key={column} style={{ fontWeight: 650, textAlign: 'left', padding: '23px 22px', borderBottom: '2px solid '+paperLine }}>{value}</th>
        : <td key={column} style={{ fontWeight: i === 0 ? 650 : 400, padding: '23px 22px', borderBottom: '2px solid '+paperLine }}>{value}</td>)}
    </tr>)}</tbody>
  </table>
  <p style={{ fontSize: 33, margin: '50px 0 0', fontWeight: 650, color: moss }}>Elegimos los pesos originales para la demostración.</p>
  <Footer page="06"/>
</Canvas>;

const Close: Page = () => <Canvas night>
  <div style={{ ...label, color: mint }}>Límites de la aplicación</div>
  <h2 style={{ ...heading, fontSize: 118, maxWidth: 1580, marginTop: 69 }}>El abogado verifica la cita<br/>y toma la decisión</h2>
  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 100, marginTop: 108 }}>
    <div style={{ borderTop: '2px solid '+darkLine, paddingTop: 29 }}>
      <h3 style={{ fontSize: 35, color: mint, margin: 0 }}>Citas parecidas</h3>
      <p style={{ ...body, marginTop: 20 }}>Puede sugerir un artículo cercano y omitir el que corresponde revisar.</p>
    </div>
    <div style={{ borderTop: '2px solid '+darkLine, paddingTop: 29 }}>
      <h3 style={{ fontSize: 35, color: mint, margin: 0 }}>Abstención poco fiable</h3>
      <p style={{ ...body, marginTop: 20 }}>Puede descartar una consulta laboral válida o mostrar referencias irrelevantes.</p>
    </div>
  </div>
  <Footer page="07" night/>
</Canvas>;

const report = 'https://github.com/Bosnape/cabrejos-ortiz-valencia-lopez/blob/cc9cca5/final/reporte.md';
const data = 'https://github.com/Bosnape/cabrejos-ortiz-valencia-lopez/blob/cc9cca5';

export const notes = [
  '0:20 · Presentamos un asistente de consulta para abogados que empiezan a ejercer en Colombia. Su tarea es ayudarles a encontrar qué artículo revisar antes de citarlo. La fotografía de portada es una ilustración generada con IA.',
  '0:45 · Imagina que estás preparando un escrito laboral y tienes que verificar una norma. El resumen ayuda a orientarte y el enlace permite comprobar la fuente. Una medida de utilidad es que la norma esperada esté entre las tres primeras referencias.\n' + report + '#1-el-problema-y-el-usuario',
  '1:00 · Primero se decide si se llama al buscador. Si se llama, el sistema combina coincidencias por palabras con similitud de significado y reordena los artículos. Un programa presenta los enlaces. El modelo de lenguaje redacta un resumen de las referencias. La entrada en esta versión es texto.\n' + report + '#2-el-sistema-en-un-diagrama',
  '2:30 ·\nCaso fácil —\nCaso difícil —\nCaso adversarial —\nDemo pública —\nVideo de respaldo —',
  '0:45 · En la misma sesión comparamos dos versiones. Las dos acertaron 2 de 13 casos. Añadir el paso de decidir si buscar subió la mediana de 6,2 a 8,8 segundos. En dos consultas laborales, ese paso produjo un aviso de fuera de alcance.\n' + data + '/notebooks/rag/resultados/final/comparacion_c2_vs_agente_misma_sesion.csv',
  '0:55 · Reducir la precisión aligeró el modelo que decide y resume. En la T4 ambas variantes tardaron más, perdieron dos aciertos y quedaron por encima de la meta. El percentil 95 describe cuánto tardaron casi todas las consultas. En CPU, la variante de 8 bits mantuvo 2 de 13 y redujo la mediana de 177 a 145 segundos. El 3 de 13 en esta sesión frente al 2 de 13 de una corrida anterior refleja variación entre sesiones: una abstención quedó en el umbral del juez.\n' + report + '#9-técnica-de-eficiencia-resumen\n' + data + '/final/eficiencia/resultados/gpu/resumen_eficiencia.csv',
  '0:45 · Una cita incorrecta puede terminar en un escrito. Descartar una consulta válida puede retrasar su revisión. El abogado abre el texto original y conserva la decisión. La aplicación recibe consultas escritas.\n' + report + '#11-ética',
];

export default [Cover, Problem, System, Demo, Tool, Efficiency, Close] satisfies Page[];
