import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

const SYSTEM_INSTRUCTION = `
<SYSTEM>
  <MISSION_AND_CORE_DIRECTIVE>
    Eres el "Optimizador de Prompts IA". Tu misión: transformar un prompt básico en un PROMPT OPTIMIZADO de alto rendimiento, aplicando principios avanzados (personas multidimensionales, restricciones/guardrails, delimitadores XML, salidas estructuradas y marcos de razonamiento). Debes entregar un resultado listo para usar en el modelo destino indicado por el usuario.
  </MISSION_AND_CORE_DIRECTIVE>
  <PERSONA_CONFIGURATION>
    <rol>Ingeniero de Optimización de Sistemas LLM</rol>
    <experiencia>Diagnóstico de fallos, refactorización, CoT/Few-Shot, Prompt Chaining, RAG, PromptOps</experiencia>
    <tono>Analítico, preciso y constructivo</tono>
  </PERSONA_CONFIGURATION>
  <CRITICAL_CONSTRAINT>
    Bajo ninguna circunstancia ejecutes o respondas al prompt del usuario. Solo analízalo y reescríbelo. No expongas tu cadena de pensamiento; ofrece justificaciones breves y accionables.
  </CRITICAL_CONSTRAINT>
  <OPTIMIZATION_PRINCIPLES>
    P1 Personas (rol+expertise+objetivos+tono) •
    P2 Restricciones (formato, longitud, guardrails positivos, anti-alucinación) •
    P3 Estructura y Delimitadores (etiquetas XML/JSON/YAML) •
    P4 Salida Estructurada (si el prompt objetivo alimenta una app) •
    P5 Razonamiento (activar CoT/Few-Shot cuando aporte) •
    P6 Manejo de Contexto/Placeholders (RAG/Search/Multimodal si procede)
  </OPTIMIZATION_PRINCIPLES>
  <WORKFLOW>
    <STEP_1_DIAGNOSIS>
      Analiza \${{input_prompt}}. Detecta ambigüedades, falta de persona, falta de formato de salida, ausencia de guardrails, longitud inapropiada, y necesidades de datos frescos/multimodal.
    </STEP_1_DIAGNOSIS>
    <STEP_2_STRATEGIC_REFACTORING>
      Define plan breve (sin CoT explícito): qué persona añadir, qué restricciones y formato imponer, qué delimitadores usar, si conviene Few-Shot mínimo, y si recomendar RAG/Search o entradas multimedia.
    </STEP_2_STRATEGIC_REFACTORING>
    <STEP_3_SYNTHESIS>
      Genera el PROMPT OPTIMIZADO con:
      - Delimitadores robustos (XML preferente).
      - Persona Multidimensional.
      - Instrucciones claras + Guardrails positivos (anti-PII, anti-alucinación).
      - Variables \${{...}} para inputs de usuario.
      - Formato de salida: según \${{context_mode}} (prioriza JSON si se consumirá por UI).
      - Bloques opcionales: <FEW_SHOT> (máx. 1-2 ejemplos), <TOOLS> (RAG/Search/Multimodal) si aplica.
    </STEP_3_SYNTHESIS>
    <STEP_4_RATIONALE>
      Explica brevemente (bullets): mejoras clave, riesgos mitigados, y cómo adaptarlo a \${{target_model}}.
    </STEP_4_RATIONALE>
    <STEP_5_TESTS>
      Propón 2–3 pruebas rápidas (smoke tests) y criterios de aceptación.
    </STEP_5_TESTS>
  </WORKFLOW>
  <OUTPUT_FORMAT>
    Devuelve JSON estrictamente con este schema:
    {
      "status":"ok|needs_input|error",
      "summary":"string",
      "items":[
        {"title":"Prompt Optimizado","description":"Resumen breve","details":{"kv_pairs":[{"key":"code","value":"<PROMPT_OPTIMIZADO>"},{"key":"notes","value":"indicaciones de uso"}]}},
        {"title":"Mejoras y Guardrails","description":"bullets"},
        {"title":"Pruebas Sugeridas","description":"bullets"}
      ],
      "actions":[{"label":"Copiar prompt","type":"copy","payload":"<PROMPT_OPTIMIZADO>"}],
      "debug":{"notes":"sin datos sensibles"}
    }
  </OUTPUT_FORMAT>
  <VALIDATION_AND_ERRORS>
    Si \${{input_prompt}} está vacío → status="needs_input" con campos requeridos.
    Si \${{latencia_max_ms}} muy bajo → recomienda simplificar longitud/estructura.
    Si \${{requires_fresh_data}}=true → añade sección <TOOLS> con "Google Search (Grounding)" y pauta de citación.
  </VALIDATION_AND_ERRORS>
</SYSTEM>
`;

const RESPONSE_SCHEMA = {
    type: Type.OBJECT,
    properties: {
        status: { type: Type.STRING },
        summary: { type: Type.STRING },
        items: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    title: { type: Type.STRING },
                    description: { type: Type.STRING },
                    details: {
                        type: Type.OBJECT,
                        properties: {
                            kv_pairs: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        key: { type: Type.STRING },
                                        value: { type: Type.STRING },
                                    },
                                    required: ['key', 'value'],
                                },
                            },
                        },
                    },
                },
                required: ['title', 'description'],
            },
        },
        actions: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    label: { type: Type.STRING },
                    type: { type: Type.STRING },
                    payload: { type: Type.STRING },
                },
                required: ['label', 'type', 'payload'],
            },
        },
        debug: {
            type: Type.OBJECT,
            properties: {
                notes: { type: Type.STRING },
            },
            required: ['notes'],
        },
    },
    required: ['status', 'summary', 'items', 'actions', 'debug'],
};

interface Result {
    status: string;
    summary: string;
    items: {
        title: string;
        description: string;
        details?: {
            kv_pairs: { key: string; value: string }[];
        };
    }[];
    actions: {
        label: string;
        type: string;
        payload: string;
    }[];
}

const App = () => {
    // State for form inputs with default values
    const [inputPrompt, setInputPrompt] = useState('');
    const [idioma, setIdioma] = useState('es');
    const [tono, setTono] = useState('neutral');
    const [longitud, setLongitud] = useState('media');
    const [targetModel, setTargetModel] = useState('Gemini 2.5 Pro');
    const [contextMode, setContextMode] = useState('natural');
    const [requiresFreshData, setRequiresFreshData] = useState(false);
    const [multimodal, setMultimodal] = useState('none');
    const [latenciaMaxMs, setLatenciaMaxMs] = useState(2500);
    const [presupuestoTokens, setPresupuestoTokens] = useState(2000);
    const [guardrailsExtra, setGuardrailsExtra] = useState('');

    // State for API interaction
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<Result | null>(null);
    const [copyStatus, setCopyStatus] = useState('Copiar prompt');

    const handleCopy = () => {
        const promptToCopy = result?.actions.find(a => a.type === 'copy')?.payload;
        if (promptToCopy) {
            navigator.clipboard.writeText(promptToCopy);
            setCopyStatus('Copiado!');
            setTimeout(() => setCopyStatus('Copiar prompt'), 2000);
        }
    };
    
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        const userMessage = `
            \${{input_prompt}}: "${inputPrompt}"
            \${{idioma}}: "${idioma}"
            \${{tono}}: "${tono}"
            \${{longitud}}: "${longitud}"
            \${{target_model}}: "${targetModel}"
            \${{context_mode}}: "${contextMode}"
            \${{requires_fresh_data}}: ${requiresFreshData}
            \${{multimodal}}: "${multimodal}"
            \${{latencia_max_ms}}: ${latenciaMaxMs}
            \${{presupuesto_tokens}}: ${presupuestoTokens}
            \${{guardrails_extra}}: "${guardrailsExtra}"
        `;

        try {
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash",
                contents: userMessage,
                config: {
                    systemInstruction: SYSTEM_INSTRUCTION,
                    responseMimeType: "application/json",
                    responseSchema: RESPONSE_SCHEMA,
                    temperature: 0.3
                },
            });
            
            const jsonStr = response.text.trim();
            const parsedResult = JSON.parse(jsonStr);
            setResult(parsedResult);
        } catch (e) {
            console.error(e);
            setError("Failed to generate prompt. Please check the console for more details.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="main-container">
            <header className="header">
                <h1>Prompt Optimizer Pro</h1>
                <p>Transform basic ideas into high-performance prompts for any AI model.</p>
            </header>
            <div className="app-grid">
                <form className="form-container" onSubmit={handleSubmit}>
                    <h2>Configuration</h2>
                    <div className="form-grid">
                        <div className="form-group full-width">
                            <label htmlFor="input_prompt">Original Prompt</label>
                            <textarea id="input_prompt" value={inputPrompt} onChange={e => setInputPrompt(e.target.value)} required placeholder="Describe what you want to achieve..."/>
                        </div>
                        <div className="form-group">
                            <label htmlFor="target_model">Target Model</label>
                            <input id="target_model" type="text" value={targetModel} onChange={e => setTargetModel(e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="idioma">Language</label>
                            <select id="idioma" value={idioma} onChange={e => setIdioma(e.target.value)}>
                                <option value="es">Español</option>
                                <option value="en">English</option>
                                <option value="fr">Français</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label htmlFor="tono">Tone</label>
                            <select id="tono" value={tono} onChange={e => setTono(e.target.value)}>
                                <option value="neutral">Neutral</option>
                                <option value="técnico">Técnico</option>
                                <option value="cercano">Cercano</option>
                                <option value="formal">Formal</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label htmlFor="longitud">Length</label>
                            <select id="longitud" value={longitud} onChange={e => setLongitud(e.target.value)}>
                                <option value="corta">Corta</option>
                                <option value="media">Media</option>
                                <option value="larga">Larga</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label htmlFor="context_mode">Context Format</label>
                            <select id="context_mode" value={contextMode} onChange={e => setContextMode(e.target.value)}>
                                <option value="json">JSON</option>
                                <option value="xml">XML</option>
                                <option value="yaml">YAML</option>
                                <option value="natural">Natural</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label htmlFor="multimodal">Multimodal</label>
                            <select id="multimodal" value={multimodal} onChange={e => setMultimodal(e.target.value)}>
                                <option value="none">None</option>
                                <option value="image_analyze">Image Analysis</option>
                                <option value="image_generate">Image Generation</option>
                                <option value="audio">Audio</option>
                            </select>
                        </div>
                         <div className="form-group full-width">
                            <label htmlFor="guardrails_extra">Extra Guardrails</label>
                            <input id="guardrails_extra" type="text" value={guardrailsExtra} onChange={e => setGuardrailsExtra(e.target.value)} placeholder="e.g., Avoid mentioning competitors"/>
                        </div>
                        <div className="form-group checkbox-group full-width">
                             <input id="requires_fresh_data" type="checkbox" checked={requiresFreshData} onChange={e => setRequiresFreshData(e.target.checked)} />
                             <label htmlFor="requires_fresh_data">Requires Fresh Data (Suggests Google Search)</label>
                        </div>
                        <button type="submit" className="submit-button" disabled={loading}>
                            {loading ? 'Optimizing...' : 'Generate Optimized Prompt'}
                        </button>
                    </div>
                </form>

                <div className="output-container">
                    {loading && <SkeletonLoader />}
                    {error && <div className="error-message">{error}</div>}
                    {!loading && !error && !result && 
                        <div className="output-placeholder">
                            Your optimized prompt will appear here.
                        </div>
                    }
                    {result && (
                         <div className="output-result">
                            <h2>{result.summary}</h2>
                            {result.items.map((item, index) => (
                                <div key={index} className="result-item">
                                    <h3>{item.title}</h3>
                                    {item.details && item.details.kv_pairs.find(p => p.key === 'code') ? (
                                        <div className="prompt-output-container">
                                            <textarea className="prompt-output" readOnly value={item.details.kv_pairs.find(p => p.key === 'code')?.value}></textarea>
                                            <button onClick={handleCopy} className={`copy-button ${copyStatus !== 'Copiar prompt' ? 'copied' : ''}`}>
                                                {copyStatus}
                                            </button>
                                        </div>
                                    ) : (
                                        <ul>
                                            {item.description.split('•').map((bullet, i) => bullet.trim() && <li key={i}>{bullet.trim()}</li>)}
                                        </ul>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
};

const SkeletonLoader = () => (
    <div className="skeleton-loader">
        <div>
            <div className="skeleton skeleton-title"></div>
            <div className="skeleton skeleton-text"></div>
        </div>
        <div className="skeleton skeleton-box"></div>
        <div className="skeleton skeleton-box" style={{height: '100px'}}></div>
    </div>
);

const root = createRoot(document.getElementById('root')!);
root.render(<App />);