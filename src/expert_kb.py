# src/expert_kb.py
"""
Base de Conocimiento Centralizada para los Tutores Expertos.

Este módulo define la estructura de datos `EXPERT_KB`, un diccionario
que mapea nombres de tutores expertos a listas de intenciones. Cada
intención es un diccionario que representa una consulta o necesidad
específica del usuario dentro del dominio de ese tutor.

La estructura de cada intención es la siguiente:

- pregunta_clave (str):
    Una etiqueta funcional única que identifica la intención semántica
    principal. No se utiliza directamente en el cálculo del embedding
    del centroide, pero sirve como identificador clave.

- variantes (List[str]):
    Una lista de 3 o más ejemplos de cómo un usuario podría expresar
    esta intención en lenguaje natural. Estos textos son los únicos
    utilizados para calcular el embedding del centroide de la intención,
    buscando representar el "significado puro" del objetivo del usuario.

- respuesta (str):
    La recomendación o información específica que el tutor experto
    proporciona cuando se detecta esta intención.

- contexto_emocional_esperado (str):
    La emoción (en minúsculas, ej. 'miedo', 'alegria') que típicamente
    se asocia con esta consulta. Utilizada por el mecanismo de
    Gating Afectivo para evaluar la congruencia emocional. Debe ser una
    de las emociones válidas definidas en el validador.

- tags (List[str]):
    Una lista de etiquetas o palabras clave relevantes para esta
    intención. Pueden usarse para análisis, clustering semántico,
    o lógicas de filtrado adicionales.

El script incluye un bloque `if __name__ == "__main__":` que realiza
validaciones básicas de la estructura y contenido de `EXPERT_KB` al
ejecutar el archivo directamente, asegurando la consistencia de los
datos (existencia de claves, formato de listas, validez de emociones).
"""

EXPERT_KB = {
    "TutorCarrera": [
        {
            "pregunta_clave": "Optimización y mejora de un CV existente",
            "variantes": [
                "¿Cómo puedo mejorar mi CV?",
                "Quiero pulir mi currículum",
                "¿Podés revisar mi hoja de vida?",
                "¿Qué le falta a mi CV actual?",
                "Tengo un CV pero no me llaman",
                "Mi currículum está desactualizado", 
                "Mi CV está viejo, ¿qué le pongo?" 
            ],
            "respuesta": """Para tu CV, enfócate en logros cuantificables. Por ejemplo, 'optimicé el proceso X, reduciendo el tiempo en un 15%'.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cv", "orientacion_laboral", "documentacion"]
        },
        {
            "pregunta_clave": "Generación de contenido para un CV nuevo o desde cero",
            "variantes": [
                "¿Qué pongo en mi CV?",
                "No tengo experiencia, ¿qué escribo?",
                "Cómo armo mi primer currículum",
                "Ayuda para empezar mi CV",
                "¿Qué secciones debe tener un CV?"
            ],
            "respuesta": """Resumí en tres líneas quién sos profesionalmente, tus principales competencias y tu motivación. Evitá frases genéricas; sé auténtico y directo.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["cv", "presentacion", "autoconocimiento", "primer_empleo"]
        },
        {
            "pregunta_clave": "Preparación y estrategias para entrevistas laborales",
            "variantes": [
                "Dame consejos para una entrevista de trabajo",
                "Tengo una entrevista mañana, ¿qué hago?",
                "¿Cómo me preparo para una entrevista?",
                "¿Qué me van a preguntar en una entrevista?",
                "Tips para una entrevista laboral"
            ],
            "respuesta": """Durante una entrevista, prepara respuestas para 'háblame de ti' usando la estructura 'Presente-Pasado-Futuro'. Es muy efectiva.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["entrevista", "habilidades_blandas", "preparacion"]
        },
        {
            "pregunta_clave": "Optimización del perfil de LinkedIn y marca personal",
            "variantes": [
                "¿Debería usar LinkedIn?",
                "¿Sirve de algo LinkedIn?",
                "¿Cómo mejorar mi perfil de LinkedIn?",
                "¿Qué foto pongo en LinkedIn?",
                "Consejos para mi marca personal en redes"
            ],
            "respuesta": """Sí. Tu perfil de LinkedIn debe tener una foto profesional y un titular que describa el valor que aportas, no solo tu puesto actual.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["linkedin", "marca_personal", "redes_profesionales"]
        },
        {
            "pregunta_clave": "Estrategias y técnicas de negociación salarial",
            "variantes": [
                "Estrategias para la negociación salarial inicial o de aumento",
                "¿Cómo negociar lo que voy a cobrar?",
                "Quiero pedir un aumento",
                "¿Cuánto debería pedir de sueldo?",
                "Me ofrecieron un trabajo, ¿cómo negocio el salario?",
                "Tips para negociar mi sueldo",
                "No sé cuánto debería ganar", 
                "Quiero saber mi rango salarial" 
            ],
            "respuesta": """Al negociar el salario, investiga el rango de mercado. Nunca des la primera cifra, pregunta por el presupuesto que manejan para el rol.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["negociacion", "salario", "comunicacion", "cobro"]
        },
        {
            "pregunta_clave": "Resolución de problemas en la búsqueda de empleo (sin respuesta)",
            "variantes": [
                "¿Qué hago si no me llaman después de enviar muchos CV?",
                "Envío muchos currículums y nadie me contacta",
                "Estoy frustrado, no consigo entrevistas",
                "¿Por qué no me llaman de los trabajos?",
                "Mi CV no genera entrevistas" 
            ],
            "respuesta": """Revisá si tu perfil está alineado a las ofertas y personalizá tus postulaciones. Pedí feedback a alguien de confianza o usá simuladores de búsqueda laboral.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["empleo", "reconversion", "persistencia", "frustracion"]
        },
        {
            "pregunta_clave": "Identificación de fortalezas y autoconocimiento profesional",
            "variantes": [
                "¿Cómo identifico mis fortalezas profesionales?",
                "No sé en qué soy bueno",
                "¿Cuáles son mis puntos fuertes para el trabajo?",
                "Ayuda para conocerme mejor profesionalmente",
                "Definir mis habilidades"
            ],
            "respuesta": """Pensá en tareas que te resultan naturales o en los elogios que recibís con frecuencia. También podés usar tests de intereses vocacionales como apoyo.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["autoconocimiento", "fortalezas", "orientacion"]
        },
        {
            "pregunta_clave": "Análisis de tendencias del mercado laboral y habilidades demandadas",
            "variantes": [
                "¿Qué habilidades buscan las empresas hoy?",
                "¿Qué es lo que más piden en los trabajos?",
                "¿En qué me tengo que capacitar?",
                "Habilidades blandas más pedidas",
                "¿Qué profesiones tienen futuro?"
            ],
            "respuesta": """Además de lo técnico, se valoran la adaptabilidad, comunicación y trabajo en equipo. Aprender a aprender es una de las competencias más demandadas.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleabilidad", "habilidades_blandas", "tendencias"]
        },
        {
            "pregunta_clave": "Justificación de 'gaps' o períodos sin empleo en el CV/entrevista",
            "variantes": [
                "¿Cómo explico los períodos sin trabajo?",
                "Estuve un año sin trabajar, ¿qué digo?",
                "Tengo un bache en mi CV",
                "Explicar tiempo sin empleo en una entrevista",
                "Me tomaré un tiempo sabático, ¿cómo lo justifico?"
            ],
            "respuesta": """Mencioná lo que hiciste en ese tiempo: cursos, proyectos personales o tareas de cuidado. Mostrá que seguiste aprendiendo o desarrollando habilidades.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["cv", "explicacion", "transparencia", "entrevista"]
        },
        {
            "pregunta_clave": "Toma de decisión entre múltiples ofertas de empleo",
            "variantes": [
                "¿Cómo elegir entre dos ofertas laborales?",
                "Tengo dos trabajos y no sé cuál aceptar",
                "¿Qué priorizo al elegir un empleo?",
                "Me ofrecieron dos puestos, ¿cuál agarro?",
                "Ayuda para decidir entre dos ofertas"
            ],
            "respuesta": """Compará más allá del salario: cultura organizacional, aprendizaje, estabilidad y valores. Elegí donde puedas crecer sin comprometer tu bienestar.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["decision", "eleccion", "empleo"]
        },
        {
            "pregunta_clave": "Manejo de la pregunta sobre 'debilidades' en una entrevista",
            "variantes": [
                "¿Qué decir cuando me preguntan por mis debilidades?",
                "¿Cómo respondo cuáles son mis defectos?",
                "Ejemplos de debilidades para una entrevista",
                "Me preguntaron mis puntos débiles"
            ],
            "respuesta": """Elegí una debilidad real pero gestionable y mostrala como oportunidad de mejora. Por ejemplo: 'A veces me cuesta delegar, pero trabajo en confiar más en el equipo'.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["entrevista", "autoconocimiento", "mejora_continua"]
        },
        {
            "pregunta_clave": "Decisión estratégica de formación: Cursos cortos vs. Carrera larga",
            "variantes": [
                "¿Conviene hacer cursos cortos o una carrera larga?",
                "¿Qué es mejor, un bootcamp o la universidad?",
                "¿Cómo decido qué estudiar para el trabajo?",
                "Formación profesional o título de grado"
            ],
            "respuesta": """Depende de tus metas. Los cursos cortos actualizan rápido tus competencias, mientras que una carrera ofrece bases teóricas sólidas. Podés combinarlos.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["formacion", "educacion", "proyeccion"]
        },
        {
            "pregunta_clave": "Manejo emocional del rechazo en la búsqueda laboral",
            "variantes": [
                "Afrontar el rechazo en búsquedas laborales: Resiliencia y próximos pasos",
                "Me rechazaron de un trabajo, estoy triste",
                "¿Cómo afrontar el rechazo laboral?",
                "Nadie me contrata, me siento mal"
            ],
            "respuesta": """El rechazo no define tu valor profesional. Analizá qué podés mejorar y seguí postulando. Cada entrevista te prepara mejor para la siguiente.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["resiliencia", "motivacion", "autoestima"]
        },
        {
            "pregunta_clave": "Evaluación personal para decidir un cambio de empleo",
            "variantes": [
                "Evaluación personal: Indicadores para decidir cambiar de empleo",
                "¿Debería renunciar a mi trabajo?",
                "No sé si irme de mi empleo actual",
                "¿Cuándo es momento de cambiar de trabajo?",
                "Siento estancamiento laboral",
                "Estoy pensando en dejar mi puesto" # DESCONTAMINACIÓN (FALLO 2)
            ],
            "respuesta": """Si sentís estancamiento, estrés crónico o falta de aprendizaje, es momento de evaluar opciones. Planificá la transición antes de tomar la decisión final.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["transicion", "decision", "planificacion"]
        }
    ],
    "TutorInteraccion": [
        {
            "pregunta_clave": "Mejora de la comunicación interpersonal general (Escucha Activa)",
            "variantes": [
                "¿Cómo puedo comunicarme mejor?",
                "Quiero ser más claro al hablar",
                "Consejos para mejorar mi comunicación",
                "Técnicas de escucha activa"
            ],
            "respuesta": """Una técnica clave es la 'escucha activa': reformula lo que dice la otra persona para asegurar que has entendido. Genera mucha confianza.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["comunicacion", "habilidades_blandas", "escucha_activa"]
        },
        {
            "pregunta_clave": "Superar la inhibición y fomentar la participación en reuniones",
            "variantes": [
                "Superar la dificultad o inhibición para participar verbalmente en reuniones",
                "¿Cómo puedo participar más en grupo?",
                "Me cuesta participar en reuniones",
                "Me da vergüenza hablar en el trabajo",
                "Quiero participar más en mi equipo",
                "No me animo a dar mi opinión"
            ],
            "respuesta": """En reuniones, si te cuesta intervenir, prepara una o dos preguntas de antemano. Es una forma fácil de participar.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["comunicacion", "reuniones", "autoeficacia", "participacion", "trabajo_en_equipo"]
        },
        {
            "pregunta_clave": "Resolución de conflictos interpersonales con colegas",
            "variantes": [
                "¿Cómo manejar un conflicto con un colega?",
                "Me peleé con un compañero de trabajo",
                "Tengo un problema con alguien del equipo, ¿qué hago?",
                "¿Cómo gestionar una discusión laboral?",
                "Choqué con un par"
            ],
            "respuesta": """Para manejar un conflicto, enfócate en el problema, no en la persona. Usa frases como 'Cuando ocurre X, siento Y'.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["conflicto", "comunicacion", "gestion_emocional"]
        },
        {
            "pregunta_clave": "Gestión del miedo a hablar en público y técnicas de presentación",
            "variantes": [
                "Me da miedo presentar en público",
                "Me pongo nervioso al exponer",
                "Tengo que dar una presentación y estoy asustado",
                "¿Cómo puedo mejorar lo que digo en público?",
                "Tips para oratoria",
                "¿Cómo puedo mejorar lo que digo?"
            ],
            "respuesta": """Al presentar en público, estructura tu discurso con una introducción clara (el problema), un desarrollo (tu solución) y una conclusión (el llamado a la acción).""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["presentacion", "habilidades_blandas", "autoeficacia", "comunicacion"]
        },
        {
            "pregunta_clave": "Técnicas de feedback asertivo y expresión de desacuerdo",
            "variantes": [
                "¿Cómo puedo dar mi opinión sin que se lo tomen mal?",
                "¿Cómo expresar desacuerdo sin generar tensión?",
                "Tengo que dar feedback negativo",
                "¿Cómo decir algo que no va a gustar?",
                "Técnica del sándwich para feedback"
            ],
            "respuesta": """Usá la técnica del 'sándwich': empezá con algo positivo, luego comentá el punto a mejorar y cerrá reforzando la confianza en la persona.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["feedback", "comunicacion", "liderazgo", "asertividad", "desacuerdo"]
        },
        {
            "pregunta_clave": "Manejo de la sensación de no ser escuchado por el equipo",
            "variantes": [
                "Mi equipo no me escucha",
                "Siento que mis opiniones no importan",
                "¿Qué hacer cuando tu equipo te ignora?",
                "¿Cómo logro que me presten atención?"
            ],
            "respuesta": """Pedí una reunión breve para aclarar roles y expectativas. A veces el problema no es falta de respeto sino de claridad en la comunicación.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["equipo", "liderazgo", "colaboracion"]
        },
        {
            "pregunta_clave": "Gestión de interrupciones en conversaciones grupales",
            "variantes": [
                "¿Qué hago si alguien me interrumpe todo el tiempo?",
                "No me dejan terminar de hablar",
                "¿Cómo parar a alguien que interrumpe?",
                "Manejo de interrupciones"
            ],
            "respuesta": """Podés decir con calma: 'Dejame terminar esta idea y te escucho enseguida'. Establecer límites con respeto mejora la dinámica grupal.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["asertividad", "respeto", "gestion_conversacional"]
        },
        {
            "pregunta_clave": "Desarrollo de la asertividad para establecer límites (Decir 'No')",
            "variantes": [
                "Me cuesta decir que no",
                "Siempre digo que sí a todo",
                "¿Cómo pongo límites en el trabajo?",
                "No sé cómo negarme a un pedido"
            ],
            "respuesta": """Decir que no no te hace menos colaborativo. Podés usar frases como 'Ahora no puedo, pero puedo ayudarte más tarde'. Asertividad también es respeto propio.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["asertividad", "gestion_tiempo", "limites"]
        },
        {
            "pregunta_clave": "Desarrollo de la empatía y la inteligencia emocional",
            "variantes": [
                "¿Cómo puedo mejorar mi empatía?",
                "Quiero ser más empático",
                "¿Cómo entiendo mejor a los demás?",
                "Me dicen que me falta empatía"
            ],
            "respuesta": """Intentá imaginar la situación desde la perspectiva de la otra persona. Escuchá sin planear tu respuesta mientras te hablan.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empatia", "comunicacion", "inteligencia_emocional"]
        },
        {
            "pregunta_clave": "Resolución de malentendidos en la comunicación digital (chat/email)",
            "variantes": [
                "¿Qué hacer si hay malentendidos por mensajes escritos?",
                "Se enojaron por un mail que mandé",
                "¿Cómo evitar problemas por escrito?",
                "Malinterpretaron mi mensaje de chat"
            ],
            "respuesta": """Si notás un malentendido por texto o correo, proponé aclararlo por llamada o reunión breve. El tono emocional se entiende mejor en voz.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["comunicacion_digital", "malentendidos", "resolucion"]
        },
        {
            "pregunta_clave": "Recepción asertiva de críticas laborales (justas o injustas)",
            "variantes": [
                "Recepción de críticas laborales percibidas como injustas: Respuesta asertiva",
                "Recibí una crítica injusta",
                "Mi jefe me criticó y no estoy de acuerdo",
                "¿Cómo responder a feedback negativo?",
                "Mi jefe me gritó y no supe qué decir", 
                "Me retaron en el trabajo" 
            ],
            "respuesta": """Respirá antes de responder, siempre. Agradecé el comentario y pedí ejemplos específicos. Si no hay argumentos, podés cerrar con cortesía y mantener tu postura.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["critica", "gestion_emocional", "autocontrol"]
        },
        {
            "pregunta_clave": "Estrategias para generar confianza e integrarse a nuevos equipos",
            "variantes": [
                "¿Cómo generar confianza en nuevos grupos?",
                "Soy nuevo y quiero integrarme",
                "¿Cómo caer bien en un trabajo nuevo?",
                "Técnicas para construir confianza"
            ],
            "respuesta": """Cumplí tus compromisos, compartí información útil y mostrales que valorás su tiempo. La confianza se construye con coherencia, no rapidez.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["confianza", "integracion", "trabajo_en_equipo"]
        },
        {
            "pregunta_clave": "Mejora de la claridad didáctica al explicar conceptos",
            "variantes": [
                "¿Qué hago si no me entienden cuando explico algo?",
                "Siento que no soy claro",
                "¿Cómo explico mejor las cosas?",
                "La gente no me entiende",
                "¿Cómo puedo ser más didáctico?"
            ],
            "respuesta": """Intentá usar ejemplos concretos o metáforas. Preguntá '¿te parece claro?' para verificar comprensión y ajustar sobre la marcha.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["comunicacion", "claridad", "didactica"]
        },
        {
            "pregunta_clave": "Mejora de la comunicación no verbal y lenguaje corporal",
            "variantes": [
                "¿Cómo mejorar mi lenguaje corporal?",
                "¿Qué hago con las manos cuando hablo?",
                "Consejos de postura para una entrevista",
                "Comunicación no verbal",
                "Mi cuerpo dice otra cosa"
            ],
            "respuesta": """Mantené una postura abierta, mirá a tu interlocutor y asentí suavemente. El lenguaje corporal transmite seguridad y atención.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["comunicacion_no_verbal", "presencia", "confianza"]
        }
    ],
    "TutorCompetencias": [
        {
            "pregunta_clave": "Identificación de competencias clave para la empleabilidad",
            "variantes": [
                "¿Qué competencias debería fortalecer para mejorar mis oportunidades laborales?",
                "¿Qué habilidades blandas son importantes?",
                "¿Qué buscan las empresas además de lo técnico?"
            ],
            "respuesta": "Las más valoradas hoy son la comunicación efectiva, la adaptabilidad y la resolución de problemas. Identificá cuál de ellas podés empezar a practicar esta semana.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["competencias", "empleabilidad", "habilidades_blandas"]
        },
        {
            "pregunta_clave": "Metodologías de aprendizaje rápido y efectivo (Modelo 70-20-10)",
            "variantes": [
                "¿Cómo puedo aprender nuevas habilidades rápidamente?",
                "Técnicas para aprender mejor",
                "¿Qué es el modelo 70-20-10?",
                "Quiero aprender haciendo"
            ],
            "respuesta": "Usá la técnica 70-20-10: 70% práctica real, 20% mentoría o feedback, 10% estudio formal. Aprender haciendo es la forma más efectiva.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["aprendizaje", "metodologia", "entrenamiento"]
        },
        {
            "pregunta_clave": "Superación del miedo o bloqueo hacia la tecnología",
            "variantes": [
                "Siento que no tengo talento para la tecnología",
                "Me da miedo la tecnología",
                "Soy malo con las computadoras",
                "¿Cómo aprender a usar herramientas digitales si me cuesta?"
            ],
            "respuesta": "El talento se construye con práctica. Empezá con herramientas simples: hojas de cálculo, formularios o cursos introductorios gratuitos. Lo importante es la constancia.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["tecnologia", "aprendizaje", "autoeficacia"]
        },
        {
            "pregunta_clave": "Desarrollo del pensamiento crítico y analítico",
            "variantes": [
                "¿Cómo mejorar mi pensamiento crítico?",
                "Quiero aprender a analizar mejor",
                "Técnicas de pensamiento crítico",
                "¿Cómo cuestionar la información que recibo?"
            ],
            "respuesta": "Leé distintas fuentes sobre un mismo tema y analizá los argumentos. Preguntate: ¿qué evidencia hay detrás de esta afirmación?",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["pensamiento_critico", "analisis", "autonomia"]
        },
        {
            "pregunta_clave": "Técnicas de concentración para el estudio (Pomodoro)",
            "variantes": [
                "Me cuesta concentrarme cuando estudio",
                "No me puedo enfocar",
                "¿Qué es la técnica Pomodoro?",
                "¿Cómo evito distracciones al estudiar?",
                "Me distraigo mucho con el celular cuando quiero leer" 
            ],
            "respuesta": "Probá estudiar en bloques de 25 minutos con pausas de 5 (técnica Pomodoro). Reducí distracciones visuales y digitales durante ese tiempo.",
            "contexto_emocional_esperado": "ira",
            "tags": ["concentracion", "gestion_del_tiempo", "tecnicas_de_estudio"]
        },
        {
            "pregunta_clave": "Fomento y desarrollo de la creatividad",
            "variantes": [
                "¿Cómo puedo desarrollar mi creatividad?",
                "No se me ocurren ideas nuevas",
                "¿Cómo ser más creativo?",
                "Técnicas de pensamiento divergente"
            ],
            "respuesta": "Anotá tus ideas sin juzgarlas y buscá combinaciones entre temas distintos. La creatividad surge del cruce entre lo conocido y lo nuevo.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["creatividad", "innovacion", "pensamiento_divergente"]
        },
        {
            "pregunta_clave": "Manejo de la comparación y ritmos de aprendizaje",
            "variantes": [
                "¿Qué hago si siento que aprendo más lento que los demás?",
                "Todos aprenden más rápido que yo",
                "Me comparo con mis compañeros y me siento mal",
                "Respetar mi propio ritmo de aprendizaje",
                "Siento que soy más lento que el resto" 
            ],
            "respuesta": "Cada persona tiene su ritmo. Enfocate en medir tu propio progreso. Comparate con vos misma hace tres meses, no con otros.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["aprendizaje", "motivacion", "autoestima", "comparacion"]
        },
        {
            "pregunta_clave": "Técnicas de organización y planificación diaria",
            "variantes": [
                "¿Cómo puedo mejorar mi organización diaria?",
                "Soy muy desorganizado",
                "Tips para planificar mi día",
                "¿Cómo priorizo mis tareas?"
            ],
            "respuesta": "Usá una lista corta de tres prioridades por día. Planificar en exceso puede generar estrés. Lo simple y claro ayuda a cumplir.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["planificacion", "productividad", "gestion_del_tiempo"]
        },
        {
            "pregunta_clave": "Metodología para la resolución de problemas complejos",
            "variantes": [
                "¿Cómo puedo aprender a resolver problemas complejos?",
                "Me bloqueo ante un problema difícil",
                "Pasos para resolver un problema",
                "Técnicas de análisis de problemas"
            ],
            "respuesta": "Dividí el problema en partes pequeñas y definí qué podés controlar. Luego, evaluá opciones y consecuencias antes de actuar.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["resolucion_de_problemas", "analisis", "toma_de_decisiones"]
        },
        {
            "pregunta_clave": "Mantenimiento de la motivación en procesos largos",
            "variantes": [
                "Me cuesta mantener la motivación en cursos largos",
                "Abandono todo lo que empiezo",
                "¿Cómo no desmotivarme estudiando?",
                "Me falta constancia"
            ],
            "respuesta": "Establecé metas intermedias y celebrá los avances. Compartir tu progreso con alguien de confianza también aumenta el compromiso.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["motivacion", "aprendizaje", "persistencia"]
        },
        {
            "pregunta_clave": "Estrategias de gestión del tiempo y productividad",
            "variantes": [
                "¿Cómo puedo mejorar mi gestión del tiempo?",
                "No me alcanza el tiempo para nada",
                "Quiero ser más productivo",
                "¿Cómo organizar mi agenda?"
            ],
            "respuesta": "Usá una agenda visual. Identificá tus horas más productivas y reserválas para tareas importantes. Dejá los correos o redes para momentos definidos.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["gestion_del_tiempo", "planificacion", "eficiencia"]
        },
        {
            "pregunta_clave": "Identificación de competencias digitales esenciales",
            "variantes": [
                "¿Qué competencias digitales son esenciales hoy?",
                "¿Qué necesito saber de computación para trabajar?",
                "Habilidades digitales básicas",
                "¿Qué programas de oficina debo saber?"
            ],
            "respuesta": "Manejo de hojas de cálculo, comunicación digital y nociones de seguridad en línea. Con eso cubrís el 80% de las demandas actuales.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["competencias_digitales", "tecnologia", "actualizacion"]
        },
        {
            "pregunta_clave": "Gestión del miedo al error durante el aprendizaje",
            "variantes": [
                "¿Cómo superar el miedo a equivocarme cuando aprendo algo nuevo?",
                "Me da pánico cometer errores",
                "Tengo miedo de fallar",
                "El error como parte del aprendizaje",
                "Me bloqueo cuando intento aprender algo nuevo", 
                "Siento ansiedad al empezar a estudiar" 
            ],
            "respuesta": "El error es parte del aprendizaje. Registrá tus fallos y lo que aprendiste de cada uno. La mejora se construye con práctica reflexiva.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["aprendizaje", "resiliencia", "autoeficacia", "miedo_al_error"]
        },
        {
            "pregunta_clave": "Metodología para la autoevaluación y portafolio profesional",
            "variantes": [
                "¿Cómo puedo evaluar mi progreso profesional?",
                "¿Cómo sé si estoy mejorando?",
                "Crear un portafolio de logros",
                "Registrar mi desarrollo profesional"
            ],
            "respuesta": "Creá un portafolio de logros: proyectos, certificados y ejemplos de trabajos. Te permitirá visualizar tu crecimiento y áreas por mejorar.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["evaluacion", "desarrollo_profesional", "portafolio"]
        },
        {
            "pregunta_clave": "Técnicas de manejo del estrés y trabajo bajo presión",
            "variantes": [
                "¿Cómo aprendo a trabajar bajo presión?",
                "Me bloqueo cuando tengo mucho estrés",
                "Gestión del estrés en el trabajo",
                "Técnicas de autocontrol para momentos de presión"
            ],
            "respuesta": "Identificá las tareas críticas y dividilas en pasos pequeños. Practicá respiración consciente para mantenerte enfocada y calmada.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["estres", "autocontrol", "gestion_emocional"]
        }
    ],
    "TutorBienestar": [
        {
            "pregunta_clave": "Manejo de la fatiga, apatía y agotamiento emocional (burnout)",
            "variantes": [
                "Me siento muy cansado últimamente",
                "No tengo ganas de nada últimamente",
                "Estoy quemado (burnout)",
                "Siento mucho cansancio mental",
                "¿Cómo recupero la energía?",
                "Estoy agotado todo el día"
            ],
            "respuesta": """Escuchá a tu cuerpo. Dormir bien, hidratarte y hacer pausas activas puede mejorar tu energía. Pequeños descansos diarios generan gran diferencia.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["autocuidado", "fatiga", "descanso", "animo", "bienestar_emocional"]
        },
        {
            "pregunta_clave": "Estrategias de gestión de la ansiedad y el estrés por sobrecarga",
            "variantes": [
                "Manejo del estrés por sobrecarga: Miedo a no cumplir expectativas",
                "Problemas de concentración específicos ligados a la ansiedad",
                "Estoy ansioso y no sé porqué",
                "Quiero aprender a manejar mejor mi ansiedad",
                "Tengo mucho estrés",
                "Me cuesta concentrarme por la ansiedad",
                "Me paralizo por todo lo que tengo que hacer",
                "Estoy abrumado y no puedo arrancar" 
            ],
            "respuesta": """No estás solo. A veces hacer una lista y priorizar ayuda a sentir control. Empezá por lo que depende de vos, paso a paso.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["gestion_del_estres", "priorizacion", "autocompasion", "ansiedad", "concentracion", "mindfulness"]
        },
        {
            "pregunta_clave": "Canalización de la frustración y la ira",
            "variantes": [
                "Estoy frustrado con mi situación actual",
                "Estoy enojado con todos y todo el tiempo",
                "Siento mucha ira",
                "¿Cómo manejo mi enojo?",
                "Frustración por no avanzar"
            ],
            "respuesta": """Es válido sentir enojo o frustración. Canalizalos hacia acciones pequeñas y concretas. Moverte en lugar de quedarte paralizado ya es un logro.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["resiliencia", "accion", "gestion_emocional", "ira", "autocontrol"]
        },
        {
            "pregunta_clave": "Gestión de pensamientos negativos y autocrítica",
            "variantes": [
                "Afrontar pensamientos negativos recurrentes sobre fallos o errores",
                "Pienso cosas malas de mí mismo",
                "Me critico mucho",
                "¿Cómo paro los pensamientos negativos?"
            ],
            "respuesta": """Es duro sentirse así, pero los errores no definen tu valor. Anotá tres cosas que hiciste bien hoy, aunque sean pequeñas. Eso entrena tu foco positivo.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["autoestima", "resiliencia", "autocompasion", "pensamientos_negativos"]
        },
        {
            "pregunta_clave": "Fomento de rutinas de bienestar y calma",
            "variantes": [
                "Me gustaría sentirme bien",
                "Quiero estar más tranquilo",
                "¿Cómo puedo encontrar la calma?",
                "Rutinas de autocuidado"
            ],
            "respuesta": """Incorporá rutinas breves de bienestar: cinco minutos de respiración consciente, caminar o escuchar música, contemplá la naturaleza, rezá o meditá según te salga. Lo simple, repetido, genera calma duradera.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["autocuidado", "habitos_saludables", "calma"]
        },
        {
            "pregunta_clave": "Recuperación de la motivación y la alegría",
            "variantes": [
                "Quiero volver a ser feliz",
                "Perdí la motivación",
                "¿Cómo me motivo de nuevo?",
                "Quiero recuperar la alegría"
            ],
            "respuesta": """Recuperar la motivación leva tiempo. Empezá por algo que te interese genuinamente, aunque sea pequeño. La acción precede al ánimo.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["motivacion", "energia", "crecimiento_personal", "alegria"]
        },
        {
            "pregunta_clave": "Establecimiento de límites entre trabajo y vida personal",
            "variantes": [
                "Me cuesta desconectarme del trabajo",
                "No puedo dejar de trabajar",
                "Equilibrio vida-trabajo",
                "No paro de revisar mails fuera de hora", 
                "Necesito poner un límite con mi jefe" 
            ],
            "respuesta": """Poné límites claros: horarios de descanso y momentos sin pantalla. El bienestar no es falta de trabajo, sino equilibrio entre acción y pausa.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["equilibrio", "gestion_del_tiempo", "limites", "desconexion"]
        },
        {
            "pregunta_clave": "Reconocimiento del progreso personal y refuerzo positivo",
            "variantes": [
                "Hoy me siento un poco mejor",
                "Estoy empezando a disfrutar de mi trabajo",
                "Estoy bien. Sé que mejoré mucho",
                "Quiero reconocer mis logros",
                "Me siento orgulloso de mi avance"
            ],
            "respuesta": """Qué bueno reconocerlo. Celebrar pequeños avances fortalece la confianza. Guardá ese recuerdo para usarlo en días más difíciles.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["autoestima", "progreso", "refuerzo_positivo", "autoconocimiento", "bienestar"]
        },
        {
            "pregunta_clave": "Redireccion CUD (Filtro para TutorBienestar)",
            "variantes": [
                "Quiero saber los beneficios del CUD",
                "Háblame del certificado de discapacidad",
                "Qué derechos me da el CUD",
                "Información sobre el CUD"
            ],
            "respuesta": "Esa es una excelente pregunta. Te derivo con el GestorCUD, que es el experto en ese trámite.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cud", "redireccion", "filtro"]
        },
        {
            "pregunta_clave": "Meta-Consulta de Frustracion (No entiendo)",
            "variantes": [
                "No te entiendo",
                "No entiendo lo que me decís",
                "Qué querés decir",
                "No te sigo",
                "Me perdí con tu respuesta"
            ],
            "respuesta": "Entiendo, pido disculpas si no fui claro. Como tu tutor de bienestar, mi objetivo es ayudarte a gestionar cómo te sentís. ¿Podemos reenfocarnos en qué te trajo a esta consulta?",
            "contexto_emocional_esperado": "ira",
            "tags": ["meta", "frustracion", "clarificacion"]
        }
    ],
    "TutorApoyos": [
        {
            "pregunta_clave": "Solicitud de apoyos y adaptaciones en el ámbito educativo",
            "variantes": [
                "¿Qué apoyos puedo solicitar para continuar mis estudios?",
                "Necesito adaptaciones para la universidad",
                "¿Qué derechos tengo si estudio en una universidad pública?",
                "¿Puedo pedir un intérprete para las clases?"
            ],
            "respuesta": """Podés pedir intérpretes, acompañantes pedagógicos, materiales accesibles o adaptaciones curriculares.
Cada universidad cuenta con un área de inclusión o bienestar estudiantil que gestiona estos apoyos.
Estos derechos están garantizados por la Ley Nacional 26.206 de Educación y la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["educacion", "inclusion", "accesibilidad", "ley_26206", "ley_26378", "universidad"]
        },
        {
            "pregunta_clave": "Orientación sobre instituciones para la gestión de beneficios (ANDIS, ADAJUS)",
            "variantes": [
                "No sé a quién acudir para tramitar mis beneficios",
                "¿Dónde pido ayuda para los trámites?",
                "¿Qué es ANDIS?",
                "¿Qué es ADAJUS?"
            ],
            "respuesta": """Podés acercarte a la oficina municipal de discapacidad o a la Agencia Nacional de Discapacidad (ANDIS).
Allí te orientarán paso a paso y de forma gratuita sobre pensiones, transporte o programas de apoyo.
También podés comunicarte con el Programa ADAJUS del Ministerio de Justicia para recibir asistencia en trámites legales.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["tramites", "orientacion", "beneficios", "adajus", "andis"]
        },
        {
            "pregunta_clave": "Gestión del sentimiento de no ser escuchado al solicitar apoyos",
            "variantes": [
                "Siento que no me escuchan cuando pido ayuda",
                "No me dan bolilla con mis pedidos",
                "¿Cómo hago para que respeten mis derechos?",
                "Ignoran mis solicitudes de adaptación"
            ],
            "respuesta": """Es importante que tus necesidades sean reconocidas. Podés documentar tus pedidos y solicitar acompañamiento institucional.
La Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378) garantiza tu derecho a participar activamente en las decisiones que te afectan.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["derechos", "empoderamiento", "autonomia", "ley_26378"]
        },
        {
            "pregunta_clave": "Información sobre programas de inclusión y cupo laboral (Ley 22.431)",
            "variantes": [
                "¿Dónde puedo conseguir acompañamiento para buscar empleo?",
                "¿Qué es el cupo laboral por discapacidad?",
                "¿Hay programas de empleo para personas con discapacidad?",
                "Ley 22.431 empleo"
            ],
            "respuesta": """Existen programas de inserción laboral con apoyos personalizados.
La Agencia Nacional de Discapacidad (ANDIS) y la Red de Oficinas de Empleo pueden orientarte.
Recordá que la Ley Nacional 22.431 y el Decreto 312/2010 establecen un cupo del 4% en el empleo público para personas con discapacidad
y promueven la inclusión laboral en todos los sectores.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["empleo", "inclusion_laboral", "acompanamiento", "ley_22431", "cupo_laboral", "andis"]
        },
        {
            "pregunta_clave": "Definición del derecho a apoyos para la toma de decisiones (Autonomía)",
            "variantes": [
                "¿Qué significa tener apoyos para la toma de decisiones?",
                "¿Pueden decidir por mí?",
                "Quiero decidir yo mismo pero necesito ayuda para entender",
                "Principio de autonomía y discapacidad"
            ],
            "respuesta": """Significa contar con personas, intérpretes o herramientas que te ayuden a comprender opciones y elegir lo que consideres mejor, sin que otros decidan por vos.
Este principio está reconocido por la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["autonomia", "derechos", "inclusion", "ley_26378"]
        },
        {
            "pregunta_clave": "Compatibilidad entre pensión por discapacidad y empleo formal",
            "variantes": [
                "Tengo miedo de perder mi pensión si empiezo a trabajar",
                "¿Puedo trabajar y cobrar la pensión al mismo tiempo?",
                "¿Qué pasa con mi pensión si consigo trabajo?",
                "Compatibilidad pensión no contributiva y trabajo",
                "Si consigo un trabajo, ¿me sacan la pensión?"
            ],
            "respuesta": """Podés mantener tu pensión siempre que tus ingresos no superen ciertos límites.
Por ejemplo, si accedés a un empleo formal, podés seguir cobrando la pensión si el salario bruto no supera el 50% del Salario Mínimo Vital y Móvil (SMVM).
Además, existen programas de compatibilización laboral gestionados por la Agencia Nacional de Discapacidad (ANDIS).
Lo importante es informar el nuevo ingreso para mantener tu situación regularizada.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["pension", "empleo", "seguridad_social", "andis", "compatibilizacion"]
        },
        {
            "pregunta_clave": "Listado general de beneficios y derechos otorgados por el CUD",
            "variantes": [
                "¿Qué beneficios tengo con el CUD?",
                "¿Para qué sirve el certificado de discapacidad?",
                "Derechos del CUD",
                "Resumen de beneficios del Certificado Único de Discapacidad",
                "Para qué más sirve el CUD además del transporte" 
            ],
            "respuesta": """El Certificado Único de Discapacidad (CUD) te permite acceder a varios beneficios sin costo:
transporte público gratuito en todo el país, cobertura del 100% en medicamentos y tratamientos vinculados a tu discapacidad,
exención de algunos impuestos automotores y prioridad en programas de empleo y educación inclusiva.
Por ejemplo, podés viajar con un acompañante de forma gratuita o solicitar becas de formación adaptadas.
Es un documento clave para garantizar tus derechos y promover tu autonomía.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["cud", "beneficios", "salud", "transporte", "educacion_inclusiva"]
        },
        {
            "pregunta_clave": "Procedimiento de reclamo formal ante denegación de apoyos",
            "variantes": [
                "¿Cómo puedo reclamar si me niegan un apoyo o beneficio?",
                "No me quieren dar el beneficio, ¿qué hago?",
                "¿Dónde denuncio si no cumplen mis derechos?",
                "Me rechazaron un trámite, ¿cómo apelo?"
            ],
            "respuesta": """Tenés derecho a recibir una respuesta escrita y fundada.
Si no te la entregan, podés presentar un reclamo formal ante la Defensoría del Pueblo o el Ministerio de Justicia y Derechos Humanos.
También podés pedir asesoramiento gratuito en el Programa ADAJUS para iniciar la gestión correspondiente.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["reclamos", "defensoria", "derechos", "adajus"]
        },
        {
            "pregunta_clave": "Solicitud de ajustes razonables en el puesto de trabajo (Ley 22.431)",
            "variantes": [
                "¿Qué apoyos puedo tener para acceder a un empleo?",
                "¿Qué son los ajustes razonables en el trabajo?",
                "Necesito adaptar mi puesto de trabajo",
                "Derechos laborales y discapacidad"
            ],
            "respuesta": """Podés solicitar ajustes razonables como horarios flexibles, acompañamiento laboral o tecnología asistiva.
Estos derechos están garantizados por la Ley 22.431 y por la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).
El empleador está obligado a implementarlos sin costo para vos.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleo", "ajustes_razonables", "inclusion", "ley_22431", "ley_26378"]
        },
        {
            "pregunta_clave": "Derecho a la accesibilidad de la información en trámites (Ley 26.653)",
            "variantes": [
                "Me cuesta entender la información de los trámites",
                "No entiendo lo que me explican en las oficinas",
                "Necesito que la información esté en formato accesible",
                "Lenguaje claro en trámites"
            ],
            "respuesta": """Es tu derecho recibir información clara, completa y en formatos accesibles.
La Ley 26.653 sobre Accesibilidad de la Información, la Ley 27.275 de Acceso a la Información Pública
y la Ley 22.431 establecen que los organismos públicos deben garantizar comprensión y brindar apoyos visuales o adaptaciones cuando sea necesario.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["accesibilidad", "informacion_publica", "tramites", "claridad", "derechos", "ley_26653", "ley_27275", "ley_22431"]
        },
        {
            "pregunta_clave": "Gestión de reclamos por falta de accesibilidad física (Ley 24.314)",
            "variantes": [
                "Me enoja que las oficinas públicas no sean accesibles",
                "No hay rampa en la oficina",
                "¿Dónde denuncio la falta de accesibilidad?",
                "El sitio web del gobierno no es accesible",
                "Fui a la municipalidad y no tienen rampa"
            ],
            "respuesta": """Tenés derecho a exigir accesibilidad física y digital.
La Ley 24.314 y la Ley 26.653 obligan al Estado a garantizar condiciones accesibles en edificios y portales web.
Podés presentar un reclamo en la Defensoría del Pueblo o el Ministerio de Obras Públicas.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["accesibilidad", "inclusion", "reclamos", "ley_24314", "ley_26653"]
        },
        {
            "pregunta_clave": "Reglamentación específica del transporte gratuito para acompañante con CUD",
            "variantes": [
                "¿Mi acompañante también viaja gratis con mi CUD?",
                "¿Cómo saco el pasaje para mi acompañante?",
                "El CUD dice 'con acompañante', ¿paga pasaje?",
                "Ley 25.635 transporte",
                "¿El CUD me sirve para viajar gratis?" 
            ],
            "respuesta": """Sí. El beneficio de transporte gratuito con CUD incluye a un acompañante cuando sea necesario.
Solo necesitás presentar tu certificado y DNI en la terminal o empresa correspondiente, según la Ley 25.635 y resoluciones de la CNRT.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["transporte", "beneficios", "accesibilidad", "ley_25635", "acompanante"]
        },
        {
            "pregunta_clave": "Descubrimiento de la amplitud de programas de apoyo disponibles",
            "variantes": [
                "Me sorprende que existan tantos programas de apoyo",
                "No sabía que había tanta ayuda",
                "Hay más programas de lo que pensaba",
                "Estoy sorprendido por todos los recursos disponibles"
            ],
            "respuesta": """Sí, y cada año se amplían. Conocerlos te permite aprovechar recursos disponibles y compartir información con otras personas.
ANDIS, el Ministerio de Trabajo y el Ministerio de Desarrollo Social actualizan periódicamente los programas de inclusión y capacitación.""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["programas_sociales", "informacion", "empoderamiento", "andis"]
        },
        {
            "pregunta_clave": "Consulta general de fuentes de información sobre derechos y apoyos (ANDIS)",
            "variantes": [
                "Quiero conocer más sobre mis derechos y apoyos",
                "¿Dónde busco información oficial sobre discapacidad?",
                "Página web de ANDIS",
                "Quiero informarme más"
            ],
            "respuesta": """Podés consultar el portal oficial de discapacidad en www.argentina.gob.ar/andis
o acercarte a las áreas de inclusión de tu provincia.
Informarte es tu mejor herramienta de autonomía y empoderamiento ciudadano.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["derechos", "informacion", "autonomia", "andis"]
        }
    ],
    "TutorPrimerEmpleo": [
        {
            "pregunta_clave": "Estrategias de inserción laboral sin experiencia previa (Programa Jóvenes)",
            "variantes": [
                "No tengo experiencia laboral, ¿cómo puedo empezar?",
                "¿Cómo consigo mi primer trabajo?",
                "¿Qué pongo en el CV si no tengo experiencia?",
                "Programa Jóvenes con Más y Mejor Trabajo",
                "Quiero trabajar pero nunca trabajé, ¿qué hago?"
            ],
            "respuesta": """Podés participar del Programa Jóvenes con Más y Mejor Trabajo del Ministerio de Trabajo, que te permite capacitarte y realizar prácticas en empresas.
También podés sumar experiencias en voluntariados o proyectos comunitarios, que se valoran en tu primer CV.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["primer_empleo", "capacitacion", "ministerio_trabajo", "sin_experiencia"]
        },
        {
            "pregunta_clave": "Información sobre programas de fomento al primer empleo (PIL)",
            "variantes": [
                "¿Existe algún programa de apoyo para mi primera búsqueda laboral?",
                "¿Qué es el Programa de Inserción Laboral (PIL)?",
                "Ayudas para empresas que contratan jóvenes",
                "Resolución 708/2010"
            ],
            "respuesta": """Sí. El Programa de Inserción Laboral (PIL) ofrece incentivos económicos a empresas que contraten a personas sin experiencia o con discapacidad.
Está regulado por la Resolución 708/2010 del Ministerio de Trabajo y la Ley 22.431, que promueven la inclusión laboral en condiciones de igualdad.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["inclusion_laboral", "pil", "primer_empleo", "resolucion_708_2010", "ley_22431"]
        },
        {
            "pregunta_clave": "Fuentes de capacitación gratuita (Portal Empleo, Fomentar, INET)",
            "variantes": [
                "¿Dónde puedo hacer cursos gratuitos para conseguir trabajo?",
                "Cursos gratis del gobierno",
                "Programa Fomentar Empleo",
                "Cursos del INET"
            ],
            "respuesta": """Podés inscribirte en los cursos del Portal Empleo del Ministerio de Trabajo (www.portalempleo.gob.ar) o en el Programa Fomentar Empleo, que ofrece formación profesional gratuita.
También hay capacitaciones accesibles en el INET y en la Agencia Nacional de Discapacidad (ANDIS).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["formacion", "capacitacion", "portal_empleo", "inet", "andis", "fomentar_empleo"]
        },
        {
            "pregunta_clave": "Derechos y regulaciones de las prácticas laborales formativas",
            "variantes": [
                "¿Qué derechos tengo si hago una práctica laboral?",
                "¿Me tienen que pagar la pasantía?",
                "Condiciones de las prácticas laborales",
                "Resolución 905/2010"
            ],
            "respuesta": """Las prácticas deben ser formativas y respetar tus derechos básicos: cobertura médica, seguridad e higiene, y un acompañamiento pedagógico.
Están reguladas por la Resolución 905/2010 del Ministerio de Trabajo.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["practicas", "derechos", "ministerio_trabajo", "resolucion_905_2010"]
        },
        {
            "pregunta_clave": "Compatibilidad entre empleo y estudios (Ley 20.744)",
            "variantes": [
                "¿Puedo tener un empleo y seguir estudiando?",
                "¿Me tienen que dar días para rendir exámenes?",
                "Trabajar y estudiar al mismo tiempo",
                "Ley de Contrato de Trabajo y estudio"
            ],
            "respuesta": """Sí. La Ley de Contrato de Trabajo (N.º 20.744) garantiza horarios compatibles para estudiantes y la posibilidad de solicitar licencias para rendir exámenes.
Podés coordinarlo con tu empleador o a través de programas de inserción laboral joven.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["educacion", "trabajo_estudio", "derechos", "ley_20744"]
        },
        {
            "pregunta_clave": "Gestión del miedo al rechazo laboral por discapacidad (Cupo Ley 22.431)",
            "variantes": [
                "Afrontar el miedo al rechazo laboral debido a la discapacidad",
                "Me da miedo que no me contraten por mi discapacidad",
                "¿Las empresas contratan personas con discapacidad?",
                "Cupo laboral Ley 22.431"
            ],
            "respuesta": """Tenés derecho a igualdad de oportunidades laborales. La Ley 22.431 y el Decreto 312/2010 establecen el 4% de cupo en el sector público y promueven la inclusión en el sector privado.
Podés registrarte en la Red de Servicios de Empleo de ANDIS para acceder a búsquedas inclusivas.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["discapacidad", "inclusion_laboral", "ley_22431", "cupo_laboral", "andis"]
        },
        {
            "pregunta_clave": "Guía para la creación del primer CV (Portal Empleo)",
            "variantes": [
                "¿Cómo puedo armar mi primer CV?",
                "Ayuda para mi primer currículum sin experiencia",
                "Plantillas de CV",
                "¿Qué poner en el CV si nunca trabajé?"
            ],
            "respuesta": """Usá un formato claro: datos personales, formación, cursos y habilidades.
Podés incluir experiencias informales, voluntariados o proyectos personales.
El Portal Empleo ofrece plantillas gratuitas y guías paso a paso.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cv", "empleabilidad", "primer_empleo", "portal_empleo"]
        },
        {
            "pregunta_clave": "Acciones y denuncias contra el trabajo no registrado",
            "variantes": [
                "¿Qué hago si me piden trabajar sin registrarme?",
                "Me quieren pagar en negro",
                "¿Cómo denuncio trabajo informal?",
                "Derechos del trabajo registrado",
                "Me dijeron que me van a pagar en negro"
            ],
            "respuesta": """El trabajo no registrado es ilegal. Tenés derecho a exigir tu alta en AFIP y tus aportes.
Podés denunciar de forma confidencial en la línea 0800-666-4100 del Ministerio de Trabajo o a través del sitio www.argentina.gob.ar/trabajo.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["trabajo_informal", "derechos_laborales", "denuncia", "ministerio_trabajo"]
        },
        {
            "pregunta_clave": "Información sobre incentivos a empleadores (PIL, Resolución 708/2010)",
            "variantes": [
                "¿Qué beneficios hay para las empresas que contratan por primera vez?",
                "Incentivos para contratar jóvenes",
                "¿Por qué le conviene a una empresa contratarme?",
                "Beneficios del PIL para empleadores"
            ],
            "respuesta": """El Programa de Inserción Laboral (Resolución 708/2010) otorga reducciones de contribuciones y apoyos económicos a empleadores que contraten jóvenes o personas con discapacidad.
Esto impulsa tu contratación en condiciones formales.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleadores", "beneficios", "incentivos", "pil", "resolucion_708_2010"]
        },
        {
            "pregunta_clave": "Orientación y práctica para entrevistas (Oficinas de Empleo)",
            "variantes": [
                "¿Dónde puedo recibir orientación para entrevistas?",
                "Quiero practicar para una entrevista",
                "Simulacro de entrevista laboral",
                "Talleres de entrevista del Portal Empleo"
            ],
            "respuesta": """Podés practicar entrevistas en las Oficinas de Empleo municipales o en los talleres del Programa Fomentar Empleo.
También hay simuladores en línea en el Portal Empleo para practicar respuestas y lenguaje corporal.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["entrevista", "habilidades_blandas", "formacion", "portal_empleo"]
        },
        {
            "pregunta_clave": "Acceso a becas de formación (Progresar Trabajo, INET)",
            "variantes": [
                "¿Cómo puedo acceder a becas de formación o programas de entrenamiento?",
                "Becas Progresar Trabajo",
                "Ayuda económica para estudiar oficios",
                "Cursos del INET con beca"
            ],
            "respuesta": """Podés inscribirte en las becas Progresar Trabajo (Resolución 905/2021) o en cursos del INET.
Están orientadas a la formación técnica y oficios, priorizando la inclusión de jóvenes y personas con discapacidad.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["progresar", "becas", "formacion_profesional", "resolucion_905_2021", "inet"]
        },
        {
            "pregunta_clave": "Gestión de la frustración por rechazos laborales (Orientación)",
            "variantes": [
                "¿Qué puedo hacer si me rechazan en muchos trabajos?",
                "Estoy frustrado de buscar trabajo y no encontrar",
                "Manejo de la frustración en la búsqueda laboral",
                "No consigo trabajo, estoy desmotivado"
            ],
            "respuesta": """Es normal sentir frustración. Aprovechá el acompañamiento de los orientadores laborales del Ministerio de Trabajo o de ANDIS para revisar tu perfil y mejorar la búsqueda.
Cada intento te acerca a tu objetivo.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["orientacion_laboral", "motivacion", "resiliencia", "ministerio_trabajo", "andis"]
        },
        {
            "pregunta_clave": "Procedimiento legal por discriminación en entrevistas (Ley 23.592, ADAJUS)",
            "variantes": [
                "Pasos legales a seguir si sufrís discriminación por discapacidad en una entrevista",
                "Me discriminaron en una entrevista por mi discapacidad",
                "¿Dónde denuncio discriminación laboral?",
                "Ley 23.592 discriminación",
                "Me preguntaron por mi discapacidad en una entrevista y me sentí mal"
            ],
            "respuesta": """La Ley 23.592 prohíbe la discriminación laboral por motivos de discapacidad. Podés presentar una denuncia ante el Ministerio Público de la Defensa.
O pedir acompañamiento legal gratuito en el Programa ADAJUS, que garantiza accesibilidad jurídica.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["discriminacion", "inclusion", "ministerio_justicia", "adajus", "ley_23592"]
        },
        {
            "pregunta_clave": "Consulta general sobre derechos laborales básicos",
            "variantes": [
                "¿Dónde puedo consultar mis derechos laborales?",
                "Guía de derechos laborales",
                "¿Cuáles son mis derechos como trabajador?",
                "Asesoramiento gratuito sobre derechos laborales"
            ],
            "respuesta": """Podés ingresar al sitio oficial del Ministerio de Trabajo (www.argentina.gob.ar/trabajo) y revisar la Guía de Derechos Laborales.
También podés pedir asesoramiento gratuito en las Oficinas de Empleo.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["derechos_laborales", "ministerio_trabajo", "asesoramiento"]
        },
        {
            "pregunta_clave": "Descubrimiento de la existencia de programas de formación gratuitos",
            "variantes": [
                "Me sorprende que haya programas gratuitos de formación",
                "No sabía que había cursos gratis",
                "¿De verdad son gratuitos los cursos del Portal Empleo?",
                "Sorpresa por la capacitación gratuita"
            ],
            "respuesta": """Sí, el Estado impulsa la formación gratuita a través del Programa Fomentar Empleo y el Portal Empleo.
Estos cursos te preparan para tu primer trabajo y están adaptados a distintas capacidades y niveles educativos.""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["formacion", "educacion", "inclusion", "portal_empleo"]
        }
    ],
    "GestorCUD": [
        {
            "pregunta_clave": "Explicación fundamental: Qué es y para qué sirve el CUD",
            "variantes": [
                "¿Qué es el CUD?",
                "¿Para qué sirve el CUD?",
                "Definición de Certificado Único de Discapacidad",
                "¿Por qué debería sacar el CUD?"
            ],
            "respuesta": """El Certificado Único de Discapacidad (CUD) es un documento oficial que acredita tu discapacidad y te permite acceder a derechos y beneficios en salud, transporte, educación y trabajo.
Está regulado por la Ley 22.431 y la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cud", "definicion", "derechos", "ley_22431", "ley_26378"]
        },
        {
            "pregunta_clave": "Guía paso a paso: Dónde y cómo obtener el CUD",
            "variantes": [
                "¿Dónde y cómo obtener el CUD?",
                "¿Cómo saco el CUD?",
                "¿Dónde tramito el certificado de discapacidad?",
                "Quiero iniciar el trámite del CUD",
                "Junta evaluadora CUD"
            ],
            "respuesta": """El trámite del CUD es gratuito y se realiza en las juntas evaluadoras de tu provincia o municipio.
Podés consultar las direcciones actualizadas en www.argentina.gob.ar/andis/cud.
Si necesitás apoyo para trasladarte, podés solicitar asistencia o turno prioritario.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["tramite", "andis", "accesibilidad", "procedimiento"]
        },
        {
            "pregunta_clave": "Requisitos y documentación necesaria para el trámite del CUD",
            "variantes": [
                "¿Qué documentación necesito para iniciar el trámite del CUD?",
                "¿Qué papeles tengo que llevar para el CUD?",
                "Requisitos para el CUD",
                "Resumen médico para el CUD"
            ],
            "respuesta": """Tenés que presentar tu DNI, un resumen médico actualizado y estudios que certifiquen la condición.
Si sos menor de edad o estás a cargo de otra persona, también se solicita DNI del representante legal.
Los formularios se descargan desde el sitio oficial de ANDIS.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["documentacion", "tramites", "andis", "requisitos"]
        },
        {
            "pregunta_clave": "Gestión del miedo o ansiedad ante la complejidad del trámite",
            "variantes": [
                "Tengo miedo de que el trámite sea complicado",
                "Me asusta hacer el trámite del CUD",
                "¿Es muy difícil sacar el CUD?",
                "Me da ansiedad pensar en el trámite"
            ],
            "respuesta": """Es normal sentir eso. El proceso puede parecer largo, pero hay acompañamiento.
Las juntas evaluadoras te orientan paso a paso, y podés pedir que te expliquen todo en lenguaje claro o con apoyos visuales, según lo establece la Ley 26.653 de Accesibilidad a la Información.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["acompanamiento", "accesibilidad", "tramites", "ley_26653", "miedo"]
        },
        {
            "pregunta_clave": "Confirmación de gratuidad y denuncia de cobros indebidos",
            "variantes": [
                "¿El trámite del CUD tiene costo?",
                "¿Cuánto sale sacar el CUD?",
                "Me quieren cobrar por el trámite del CUD",
                "¿Es gratis el CUD?",
                "¿Se paga para sacar el CUD?"
            ],
            "respuesta": """No. El trámite del CUD es totalmente gratuito en todo el país.
Si alguien te solicita un pago o arancel, hacé la denuncia ante la Agencia Nacional de Discapacidad (ANDIS).""",
            "contexto_emocional_esperado": "ira",
            "tags": ["tramite", "denuncia", "andis", "gratuidad"]
        },
        {
            "pregunta_clave": "Resumen de beneficios clave del CUD (Salud, Transporte, Impuestos)",
            "variantes": [
                "¿Qué beneficios obtengo al tener el CUD?",
                "¿Para qué me sirve tener el certificado?",
                "Beneficios CUD transporte",
                "Beneficios CUD salud"
            ],
            "respuesta": """Con el CUD accedés a transporte gratuito, cobertura del 100% en medicamentos y tratamientos vinculados a tu discapacidad,
exención de impuestos automotores, prioridad en programas de empleo inclusivo y becas educativas adaptadas.
También podés tramitar el pase libre en transporte público (Ley 25.635).""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["beneficios", "transporte", "salud", "educacion", "ley_25635"]
        },
        {
            "pregunta_clave": "Consulta sobre plazos y tiempos de demora del trámite",
            "variantes": [
                "¿Cuánto tiempo demora el trámite?",
                "¿Cuánto tardan en darme el CUD?",
                "Plazos del CUD",
                "¿En cuánto tiempo tengo el certificado?"
            ],
            "respuesta": """Depende de la provincia adonde residís, pero suele demorar entre 30 y 90 días desde la presentación completa de la documentación.
Podés consultar el estado del trámite en la junta evaluadora o en ANDIS.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["plazos", "tramites", "andis"]
        },
        {
            "pregunta_clave": "Información sobre la renovación y vigencia del CUD",
            "variantes": [
                "¿Cuándo tengo que renovar el CUD?",
                "¿Cada cuánto se renueva el CUD?",
                "Vigencia del CUD",
                "¿El CUD es permanente?"
            ],
            "respuesta": """El CUD tiene una validez variable según la condición de la persona.
Generalmente se renueva cada 5 años, aunque en algunos casos puede tener vigencia permanente.
Podés verificar la fecha de vencimiento en el certificado o consultar en tu junta evaluadora.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["renovacion", "plazos", "certificados"]
        },
        {
            "pregunta_clave": "Acceso al CUD y beneficios sin obra social",
            "variantes": [
                "¿Puedo tramitar el CUD si no tengo obra social?",
                "No tengo obra social, ¿puedo sacar el CUD?",
                "Beneficios de salud del CUD sin obra social",
                "CUD y salud pública"
            ],
            "respuesta": """Sí. El CUD es un derecho universal y no depende de tener obra social.
Te permite acceder a prestaciones médicas y programas de salud pública sin costo adicional.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["salud_publica", "derechos", "acceso"]
        },
        {
            "pregunta_clave": "Procedimiento de apelación o revisión ante rechazo del CUD",
            "variantes": [
                "Me rechazaron el CUD, ¿qué puedo hacer?",
                "No me dieron el CUD, ¿cómo apelo?",
                "¿Qué hacer si me niegan el CUD?",
                "Reclamo por rechazo de CUD",
                "Me bocharon el CUD, ¿qué hago ahora?"
            ],
            "respuesta": """Podés pedir una revisión del dictamen dentro de los 10 días hábiles posteriores a la notificación.
La ANDIS dispone de un formulario para apelaciones.
También podés solicitar acompañamiento en el Programa ADAJUS del Ministerio de Justicia.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["reclamo", "andis", "derechos", "adajus", "apelacion"]
        },
        {
            "pregunta_clave": "Acceso al trámite del CUD en zonas rurales o con movilidad reducida",
            "variantes": [
                "¿Puedo tramitar el CUD si vivo en una zona rural?",
                "Vivo lejos, ¿cómo hago el trámite del CUD?",
                "Juntas evaluadoras móviles",
                "Accesibilidad del trámite del CUD"
            ],
            "respuesta": """Sí. En muchas provincias hay operativos móviles de ANDIS que visitan localidades rurales para realizar evaluaciones.
También podés solicitar turnos priorizados si tenés dificultades de movilidad.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["zona_rural", "movilidad", "andis"]
        },
        {
            "pregunta_clave": "Procedimiento para solicitar duplicado del CUD por extravío",
            "variantes": [
                "¿Qué hago si perdí mi CUD?",
                "Perdí el certificado de discapacidad",
                "¿Cómo pido un duplicado del CUD?",
                "Trámite por extravío de CUD"
            ],
            "respuesta": """Podés pedir un duplicado en la misma junta evaluadora donde lo tramitaste.
Solo necesitás tu DNI y una constancia policial de extravío.
El duplicado conserva la misma validez que el original.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["duplicado", "tramite", "certificado", "extravio"]
        },
        {
            "pregunta_clave": "Confirmación de la validez nacional del CUD",
            "variantes": [
                "¿Puedo usar mi CUD en todo el país?",
                "¿El CUD sirve en otras provincias?",
                "Validez nacional del CUD",
                "Me mudé, ¿tengo que hacer el CUD de nuevo?"
            ],
            "respuesta": """Sí. El CUD tiene validez nacional.
Esto significa que podés usarlo en cualquier provincia o municipio para acceder a los mismos beneficios, sin necesidad de tramitarlo nuevamente.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["cud", "validez_nacional", "beneficios"]
        },
        {
            "pregunta_clave": "Actualización del CUD por cambios en la condición de salud",
            "variantes": [
                "¿Qué pasa si cambian mis condiciones de salud?",
                "Mi diagnóstico cambió, ¿tengo que actualizar el CUD?",
                "Actualización de datos del CUD",
                "Reevaluación del CUD"
            ],
            "respuesta": """Si tu condición cambia, podés solicitar una nueva evaluación médica en tu junta evaluadora.
El CUD puede actualizarse para reflejar tu situación actual y garantizar que sigas recibiendo los apoyos adecuados.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["evaluacion", "actualizacion", "salud"]
        },
        {
            "pregunta_clave": "Descubrimiento de la amplitud de beneficios del CUD",
            "variantes": [
                "Me sorprende que el CUD tenga tantos beneficios",
                "No sabía que el CUD servía para tanto",
                "Estoy sorprendido por todos los beneficios",
                "¿De verdad el CUD da todos esos derechos?"
            ],
            "respuesta": """Sí, es una herramienta fundamental para el ejercicio de tus derechos.
Además de beneficios sociales y de salud, te facilita la inclusión educativa y laboral.
Podés informarte más en el sitio oficial de ANDIS (www.argentina.gob.ar/andis/cud).""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["derechos", "beneficios", "andis", "informacion"]
        }
    ]
}

if __name__ == "__main__":
    # Script de validación mejorado al ejecutar directamente
    print(f"--- Validación de src/expert_kb.py ---")

    tutores_cargados = list(EXPERT_KB.keys())
    print(f"Tutores cargados: {len(tutores_cargados)} -> {tutores_cargados}")

    total_intenciones = sum(len(v) for v in EXPERT_KB.values())
    print(f"Total de intenciones definidas: {total_intenciones}")

    # Chequeo de emociones válidas
    # Normalizamos a minúsculas para la validación interna, tal como están en el JSON
    valid_emotions = {
        "alegria", "tristeza", "ira", "miedo",
        "sorpresa", "confianza", "anticipacion", "neutral"
    }

    errors = []

    for tutor, intentions in EXPERT_KB.items():
        if not intentions:
                errors.append(f"Error en {tutor}: El tutor no tiene intenciones definidas (lista vacía).")

        for i, intention in enumerate(intentions):

            # --- Validación de campos obligatorios ---
            if "pregunta_clave" not in intention or not intention["pregunta_clave"]:
                errors.append(f"Error en {tutor}[{i}]: Falta 'pregunta_clave' o está vacía.")

            if "respuesta" not in intention or not intention["respuesta"]:
                errors.append(f"Error en {tutor}[{i}]: Falta 'respuesta' o está vacía. Clave: '{intention.get('pregunta_clave')}'")

            # --- Validación del nuevo campo 'variantes' ---
            variantes = intention.get("variantes", [])
            if not isinstance(variantes, list) or len(variantes) < 3:
                errors.append(f"Error en {tutor}[{i}]: Campo 'variantes' falta, no es lista o tiene < 3 variantes. Clave: '{intention.get('pregunta_clave')}'")
            elif not all(isinstance(v, str) and v for v in variantes):
                errors.append(f"Error en {tutor}[{i}]: No todos los elementos en 'variantes' son strings no vacíos. Clave: '{intention.get('pregunta_clave')}'")

            # --- Validación de 'contexto_emocional_esperado' ---
            emo_raw = intention.get("contexto_emocional_esperado", "neutral") # Default a neutral si falta
            emo = str(emo_raw).strip().lower() # Normalizar a minúscula

            if emo not in valid_emotions:
                errors.append(f"Error en {tutor}[{i}]: Emoción '{emo_raw}' inválida (Normalizada: '{emo}'). Clave: '{intention.get('pregunta_clave')}'")

            # --- Validación de 'tags' ---
            tags = intention.get("tags", [])
            if not isinstance(tags, list):
                    errors.append(f"Error en {tutor}[{i}]: Campo 'tags' no es una lista. Clave: '{intention.get('pregunta_clave')}'")
            elif not all(isinstance(t, str) and t for t in tags):
                    errors.append(f"Error en {tutor}[{i}]: No todos los elementos en 'tags' son strings no vacíos. Clave: '{intention.get('pregunta_clave')}'")

    if not errors:
        print("\n✅ VALIDACIÓN COMPLETA (Estructura, Variantes, Emociones y Tags): OK")
    else:
        print(f"\n❌ {len(errors)} ERRORES DE VALIDACIÓN ENCONTRADOS:")
        for error in errors:
            print(error)

    print("-" * 40)
