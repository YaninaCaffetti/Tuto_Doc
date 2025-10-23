# src/expert_kb.py
"""
Base de Conocimiento Centralizada para los Tutores Expertos.

Define la estructura de datos enriquecida para cada intención, incluyendo:
- pregunta_clave: La intención semántica principal (usada para el embedding).
- respuesta: La recomendación específica del tutor.
- contexto_emocional_esperado: La emoción que típicamente se asocia con esta
  consulta, usada para el "gating afectivo".
- tags: Etiquetas para clustering semántico.
"""

EXPERT_KB = {
    "TutorCarrera": [
        {
            "pregunta_clave": "¿Cómo puedo mejorar mi CV?",
            "respuesta": "Para tu CV, enfócate en logros cuantificables. Por ejemplo, 'optimicé el proceso X, reduciendo el tiempo en un 15%'.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cv", "orientacion_laboral", "documentacion"]
        },
        {
            "pregunta_clave": "Dame consejos para una entrevista de trabajo",
            "respuesta": "Durante una entrevista, prepara respuestas para 'háblame de ti' usando la estructura 'Presente-Pasado-Futuro'. Es muy efectiva.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["entrevista", "habilidades_blandas", "preparacion"]
        },
        {
            "pregunta_clave": "¿Debería usar LinkedIn?",
            "respuesta": "Sí. Tu perfil de LinkedIn debe tener una foto profesional y un titular que describa el valor que aportas, no solo tu puesto actual.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["linkedin", "marca_personal", "redes_profesionales"]
        },
        {
            "pregunta_clave": "¿Cómo negociar mi salario?",
            "respuesta": "Al negociar el salario, investiga el rango de mercado. Nunca des la primera cifra, pregunta por el presupuesto que manejan para el rol.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["negociacion", "salario", "comunicacion"]
        },
        {
            "pregunta_clave": "¿Cómo negociar lo que voy a cobrar?",
            "respuesta": "Investigá los rangos salariales del sector y esperá que el empleador proponga primero. Argumentá con base en tus competencias y logros, no en necesidad económica.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["negociacion", "salario","cobro", "comunicacion"]
        },
        {
            "pregunta_clave": "¿Qué hago si no me llaman después de enviar muchos CV?",
            "respuesta": "Revisá si tu perfil está alineado a las ofertas y personalizá tus postulaciones. Pedí feedback a alguien de confianza o usá simuladores de búsqueda laboral.",
            "contexto_emocional_esperado": "ira",
            "tags": ["empleo", "reconversion", "persistencia"]
        },
        {
            "pregunta_clave": "¿Qué pongo en mi CV?",
            "respuesta": "Resumí en tres líneas quién sos profesionalmente, tus principales competencias y tu motivación. Evitá frases genéricas; sé auténtico y directo.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["cv", "presentacion", "autoconocimiento"]
        },
        {
            "pregunta_clave": "¿Cómo puedo mejorar lo que digo?",
            "respuesta": "Practicá en voz alta, grabate y escuchá tu tono. Usá frases cortas, pausas naturales y ejemplos concretos. La claridad comunica más que la velocidad.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["comunicacion", "presentacion", "autoeficacia"]
        },
        {
            "pregunta_clave": "¿Cómo identifico mis fortalezas profesionales?",
            "respuesta": "Pensá en tareas que te resultan naturales o en los elogios que recibís con frecuencia. También podés usar tests de intereses vocacionales como apoyo.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["autoconocimiento", "fortalezas", "orientacion"]
        },
        {
            "pregunta_clave": "¿Qué habilidades buscan las empresas hoy?",
            "respuesta": "Además de lo técnico, se valoran la adaptabilidad, comunicación y trabajo en equipo. Aprender a aprender es una de las competencias más demandadas.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleabilidad", "habilidades_blandas", "tendencias"]
        },
        {
            "pregunta_clave": "¿Cómo explico los períodos sin trabajo?",
            "respuesta": "Mencioná lo que hiciste en ese tiempo: cursos, proyectos personales o tareas de cuidado. Mostrá que seguiste aprendiendo o desarrollando habilidades.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["cv", "explicacion", "transparencia"]
        },
        {
            "pregunta_clave": "¿Cómo elegir entre dos ofertas laborales?",
            "respuesta": "Compará más allá del salario: cultura organizacional, aprendizaje, estabilidad y valores. Elegí donde puedas crecer sin comprometer tu bienestar.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["decision", "eleccion", "empleo"]
        },
        {
            "pregunta_clave": "¿Qué decir cuando me preguntan por mis debilidades?",
            "respuesta": "Elegí una debilidad real pero gestionable y mostrala como oportunidad de mejora. Por ejemplo: 'A veces me cuesta delegar, pero trabajo en confiar más en el equipo'.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["entrevista", "autoconocimiento", "mejora_continua"]
        },
        {
            "pregunta_clave": "¿Conviene hacer cursos cortos o una carrera larga?",
            "respuesta": "Depende de tus metas. Los cursos cortos actualizan rápido tus competencias, mientras que una carrera ofrece bases teóricas sólidas. Podés combinarlos.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["formacion", "educacion", "proyeccion"]
        },
        {
            "pregunta_clave": "¿Cómo manejar el rechazo laboral?",
            "respuesta": "El rechazo no define tu valor profesional. Analizá qué podés mejorar y seguí postulando. Cada entrevista te prepara mejor para la siguiente.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["resiliencia", "motivacion", "autoestima"]
        },
        {
            "pregunta_clave": "¿Cómo saber si debo cambiar de trabajo?",
            "respuesta": "Si sentís estancamiento, estrés crónico o falta de aprendizaje, es momento de evaluar opciones. Planificá la transición antes de tomar la decisión final.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["transicion", "decision", "planificacion"]
        }
    ],

    "TutorInteraccion": [
        {
            "pregunta_clave": "¿Cómo puedo comunicarme mejor?",
            "respuesta": "Una técnica clave es la 'escucha activa': reformula lo que dice la otra persona para asegurar que has entendido. Genera mucha confianza.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["comunicacion", "habilidades_blandas", "escucha_activa"]
        },
        {
            "pregunta_clave": "Me cuesta hablar en reuniones",
            "respuesta": "En reuniones, si te cuesta intervenir, prepara una o dos preguntas de antemano. Es una forma fácil de participar.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["comunicacion", "reuniones", "autoeficacia"]
        },
        {
            "pregunta_clave": "¿Cómo manejar un conflicto con un colega?",
            "respuesta": "Para manejar un conflicto, enfócate en el problema, no en la persona. Usa frases como 'Cuando ocurre X, siento Y'.",
            "contexto_emocional_esperado": "ira",
            "tags": ["conflicto", "comunicacion", "gestion_emocional"]
        },
        {
            "pregunta_clave": "Me da miedo presentar en público",
            "respuesta": "Al presentar en público, estructura tu discurso con una introducción clara (el problema), un desarrollo (tu solución) y una conclusión (el llamado a la acción).",
            "contexto_emocional_esperado": "miedo",
            "tags": ["presentacion", "habilidades_blandas", "autoeficacia"]
        },
        {
            "pregunta_clave": "¿Cómo puedo dar mi opinión sin que se lo tomen mal?",
            "respuesta": "Usá la técnica del 'sándwich': empezá con algo positivo, luego comentá el punto a mejorar y cerrá reforzando la confianza en la persona.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["feedback", "comunicacion", "liderazgo"]
        },
        {
            "pregunta_clave": "Mi equipo no me escucha",
            "respuesta": "Pedí una reunión breve para aclarar roles y expectativas. A veces el problema no es falta de respeto sino de claridad en la comunicación.",
            "contexto_emocional_esperado": "ira",
            "tags": ["equipo", "liderazgo", "colaboracion"]
        },
        {
            "pregunta_clave": "¿Qué hago si alguien me interrumpe todo el tiempo?",
            "respuesta": "Podés decir con calma: 'Dejame terminar esta idea y te escucho enseguida'. Establecer límites con respeto mejora la dinámica grupal.",
            "contexto_emocional_esperado": "ira",
            "tags": ["asertividad", "respeto", "gestion_conversacional"]
        },
        {
            "pregunta_clave": "¿Cómo puedo participar más en grupo?",
            "respuesta": "Empieza con pequeños aportes: resumir ideas, proponer ejemplos o hacer preguntas. Con el tiempo ganarás presencia sin forzarte a hablar demasiado.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["trabajo_en_equipo", "participacion", "autoeficacia"]
        },
        {
            "pregunta_clave": "Me cuesta decir que no",
            "respuesta": "Decir que no no te hace menos colaborativo. Podés usar frases como 'Ahora no puedo, pero puedo ayudarte más tarde'. Asertividad también es respeto propio.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["asertividad", "gestion_tiempo", "limites"]
        },
        {
            "pregunta_clave": "¿Cómo puedo mejorar mi empatía?",
            "respuesta": "Intentá imaginar la situación desde la perspectiva de la otra persona. Escuchá sin planear tu respuesta mientras te hablan.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empatia", "comunicacion", "inteligencia_emocional"]
        },
        {
            "pregunta_clave": "¿Qué hacer si hay malentendidos por mensajes escritos?",
            "respuesta": "Si notás un malentendido por texto o correo, proponé aclararlo por llamada o reunión breve. El tono emocional se entiende mejor en voz.",
            "contexto_emocional_esperado": "ira",
            "tags": ["comunicacion_digital", "malentendidos", "resolucion"]
        },
        {
            "pregunta_clave": "¿Cómo manejar una crítica injusta?",
            "respuesta": "Respirá antes de responder, siempre. Agradecé el comentario y pedí ejemplos específicos. Si no hay argumentos, podés cerrar con cortesía y mantener tu postura.",
            "contexto_emocional_esperado": "ira",
            "tags": ["critica", "gestion_emocional", "autocontrol"]
        },
        {
            "pregunta_clave": "¿Cómo expresar desacuerdo sin generar tensión?",
            "respuesta": "Usá frases como 'Entiendo tu punto, aunque veo otra perspectiva'. Evitá el 'pero', reemplazalo por 'aunque'. Cambia el tono de la conversación.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["asertividad", "desacuerdo", "dialogo"]
        },
        {
            "pregunta_clave": "¿Cómo generar confianza en nuevos grupos?",
            "respuesta": "Cumplí tus compromisos, compartí información útil y mostrales que valorás su tiempo. La confianza se construye con coherencia, no rapidez.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["confianza", "integracion", "trabajo_en_equipo"]
        },
         {
            "pregunta_clave": "¿Cómo puedo participar más en equipo?",
            "respuesta": "Ofrecé ayuda en pequeñas tareas o proponé ideas para mejorar procesos. Participar también es escuchar y conectar con las necesidades del grupo.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["trabajo_en_equipo", "colaboracion", "participacion"]
        },
        {
            "pregunta_clave": "¿Qué hago si no me entienden cuando explico algo?",
            "respuesta": "Intentá usar ejemplos concretos o metáforas. Preguntá '¿te parece claro?' para verificar comprensión y ajustar sobre la marcha.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["comunicacion", "claridad", "didactica"]
        },
         {
            "pregunta_clave": "¿Cómo mejorar mi lenguaje corporal?",
            "respuesta": "Mantené una postura abierta, mirá a tu interlocutor y asentí suavemente. El lenguaje corporal transmite seguridad y atención.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["comunicacion_no_verbal", "presencia", "confianza"]
        },
    ],

    "TutorCompetencias": [
         {
            "pregunta_clave": "¿Qué competencias debería fortalecer para mejorar mis oportunidades laborales?",
            "respuesta": "Las más valoradas hoy son la comunicación efectiva, la adaptabilidad y la resolución de problemas. Identificá cuál de ellas podés empezar a practicar esta semana.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["competencias", "empleabilidad", "habilidades_blandas"]
        },
         {
            "pregunta_clave": "¿Cómo puedo aprender nuevas habilidades rápidamente?",
            "respuesta": "Usá la técnica 70-20-10: 70% práctica real, 20% mentoría o feedback, 10% estudio formal. Aprender haciendo es la forma más efectiva.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["aprendizaje", "metodologia", "entrenamiento"]
        },
         {
            "pregunta_clave": "Siento que no tengo talento para la tecnología",
            "respuesta": "El talento se construye con práctica. Empezá con herramientas simples: hojas de cálculo, formularios o cursos introductorios gratuitos. Lo importante es la constancia.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["tecnologia", "aprendizaje", "autoeficacia"]
        },
         {
            "pregunta_clave": "¿Cómo mejorar mi pensamiento crítico?",
            "respuesta": "Leé distintas fuentes sobre un mismo tema y analizá los argumentos. Preguntate: ¿qué evidencia hay detrás de esta afirmación?",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["pensamiento_critico", "analisis", "autonomia"]
        },
         {
            "pregunta_clave": "Me cuesta concentrarme cuando estudio",
            "respuesta": "Probá estudiar en bloques de 25 minutos con pausas de 5 (técnica Pomodoro). Reducí distracciones visuales y digitales durante ese tiempo.",
            "contexto_emocional_esperado": "ira",
            "tags": ["concentracion", "gestion_del_tiempo", "tecnicas_de_estudio"]
        },
         {
            "pregunta_clave": "¿Cómo puedo desarrollar mi creatividad?",
            "respuesta": "Anotá tus ideas sin juzgarlas y buscá combinaciones entre temas distintos. La creatividad surge del cruce entre lo conocido y lo nuevo.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["creatividad", "innovacion", "pensamiento_divergente"]
        },
         {
            "pregunta_clave": "¿Qué hago si siento que aprendo más lento que los demás?",
            "respuesta": "Cada persona tiene su ritmo. Enfocate en medir tu propio progreso. Comparate con vos misma hace tres meses, no con otros.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["aprendizaje", "motivacion", "autoestima"]
        },
         {
            "pregunta_clave": "¿Cómo puedo mejorar mi organización diaria?",
            "respuesta": "Usá una lista corta de tres prioridades por día. Planificar en exceso puede generar estrés. Lo simple y claro ayuda a cumplir.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["planificacion", "productividad", "gestion_del_tiempo"]
        },
         {
            "pregunta_clave": "¿Cómo puedo aprender a resolver problemas complejos?",
            "respuesta": "Dividí el problema en partes pequeñas y definí qué podés controlar. Luego, evaluá opciones y consecuencias antes de actuar.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["resolucion_de_problemas", "analisis", "toma_de_decisiones"]
        },
         {
            "pregunta_clave": "Me cuesta mantener la motivación en cursos largos",
            "respuesta": "Establecé metas intermedias y celebrá los avances. Compartir tu progreso con alguien de confianza también aumenta el compromiso.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["motivacion", "aprendizaje", "persistencia"]
        },
         {
            "pregunta_clave": "¿Cómo puedo mejorar mi gestión del tiempo?",
            "respuesta": "Usá una agenda visual. Identificá tus horas más productivas y reserválas para tareas importantes. Dejá los correos o redes para momentos definidos.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["gestion_del_tiempo", "planificacion", "eficiencia"]
        },
         {
            "pregunta_clave": "¿Qué competencias digitales son esenciales hoy?",
            "respuesta": "Manejo de hojas de cálculo, comunicación digital y nociones de seguridad en línea. Con eso cubrís el 80% de las demandas actuales.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["competencias_digitales", "tecnologia", "actualizacion"]
        },
         {
            "pregunta_clave": "¿Cómo superar el miedo a equivocarme cuando aprendo algo nuevo?",
            "respuesta": "El error es parte del aprendizaje. Registrá tus fallos y lo que aprendiste de cada uno. La mejora se construye con práctica reflexiva.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["aprendizaje", "resiliencia", "autoeficacia"]
        },
         {
            "pregunta_clave": "¿Cómo puedo evaluar mi progreso profesional?",
            "respuesta": "Creá un portafolio de logros: proyectos, certificados y ejemplos de trabajos. Te permitirá visualizar tu crecimiento y áreas por mejorar.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["evaluacion", "desarrollo_profesional", "portafolio"]
        },
         {
            "pregunta_clave": "¿Cómo aprendo a trabajar bajo presión?",
            "respuesta": "Identificá las tareas críticas y dividilas en pasos pequeños. Practicá respiración consciente para mantenerte enfocada y calmada.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["estres", "autocontrol", "gestion_emocional"]
        }
    ],

    "TutorBienestar": [
         {
            "pregunta_clave": "Me siento muy cansado últimamente",
            "respuesta": "Escuchá a tu cuerpo. Dormir bien, hidratarte y hacer pausas activas puede mejorar tu energía. Pequeños descansos diarios generan gran diferencia.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["autocuidado", "fatiga", "descanso"]
        },
         {
            "pregunta_clave": "Tengo miedo de no poder con todo",
            "respuesta": "No estás solo. A veces hacer una lista y priorizar ayuda a sentir control. Empezá por lo que depende de vos, paso a paso.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["gestion_del_estres", "priorizacion", "autocompasion"]
        },
         {
            "pregunta_clave": "Estoy frustrado con mi situación actual",
            "respuesta": "Es válido sentir enojo o frustración. Canalizalos hacia acciones pequeñas y concretas. Moverte en lugar de quedarte paralizado ya es un logro.",
            "contexto_emocional_esperado": "ira",
            "tags": ["resiliencia", "accion", "gestion_emocional"]
        },
         {
            "pregunta_clave": "No tengo ganas de nada últimamente",
            "respuesta": "Puede ser un signo de agotamiento emocional. Permitite descansar y hacer algo que solías disfrutar, aunque sea por unos minutos.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["animo", "bienestar_emocional", "recuperacion"]
        },
         {
            "pregunta_clave": "Me cuesta concentrarme y todo asusta",
            "respuesta": "Cuando la mente se satura, simplificar ayuda. Ordená el entorno y empezá con tareas pequeñas. Celebrá cada avance, por mínimo que sea.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["estres", "gestion_cognitiva", "autoeficacia"]
        },
         {
            "pregunta_clave": "Siento que todo me sale mal",
            "respuesta": "Es duro sentirse así, pero los errores no definen tu valor. Anotá tres cosas que hiciste bien hoy, aunque sean pequeñas. Eso entrena tu foco positivo.",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["autoestima", "resiliencia", "autocompasion"]
        },
         {
            "pregunta_clave": "Estoy enojado con todos y todo el tiempo",
            "respuesta": "La ira suele esconder cansancio o miedo. Identificá cuándo aparece y practicá respiración lenta: inhalá 4 segundos, exhalá 6. Calma el cuerpo y la mente.",
            "contexto_emocional_esperado": "ira",
            "tags": ["ira", "gestion_emocional", "autocontrol"]
        },
         {
            "pregunta_clave": "Me gustaría sentirme bien",
            "respuesta": "Incorporá rutinas breves de bienestar: cinco minutos de respiración consciente, caminar o escuchar música, contemplá la naturaleza, rezá o meditá según te salga. Lo simple, repetido, genera calma duradera.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["autocuidado", "habitos_saludables", "calma"]
        },
         {
            "pregunta_clave": "Quiero volver a ser felíz",
            "respuesta": "Recuperar la motivación lleva tiempo. Empezá por algo que te interese genuinamente, aunque sea pequeño. La acción precede al ánimo.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["motivacion", "energia", "crecimiento_personal"]
        },
         {
            "pregunta_clave": "Me cuesta desconectarme del trabajo",
            "respuesta": "Poné límites claros: horarios de descanso y momentos sin pantalla. El bienestar no es falta de trabajo, sino equilibrio entre acción y pausa.",
            "contexto_emocional_esperado": "confianza",
            "tags": ["equilibrio", "gestion_del_tiempo", "limites"]
        },
         {
            "pregunta_clave": "Estoy ansioso y no sé porqué",
            "respuesta": "La ansiedad suele anticipar más de lo que ocurre. Probá registrar tus pensamientos y reemplazarlos por afirmaciones más realistas.",
            "contexto_emocional_esperado": "miedo",
            "tags": ["ansiedad", "gestion_emocional", "autoconciencia"]
        },
         {
            "pregunta_clave": "Hoy me siento un poco mejor",
            "respuesta": "Qué bueno reconocerlo. Celebrar pequeños avances fortalece la confianza. Guardá ese recuerdo para usarlo en días más difíciles.",
            "contexto_emocional_esperado": "alegria",
            "tags": ["autoestima", "progreso", "refuerzo_positivo"]
        },
         {
            "pregunta_clave": "Estoy empezando a disfrutar de mi trabajo",
            "respuesta": "Excelente señal. Mantener lo que te hace bien consolida la mejora. Sumá variedad, pero sin exigencias: el bienestar también es equilibrio.",
            "contexto_emocional_esperado": "alegria",
            "tags": ["bienestar", "rutinas_saludables", "motivacion"]
        },
         {
            "pregunta_clave": "Quiero aprender a manejar mejor mi ansiedad",
            "respuesta": "La respiración diafragmática y los ejercicios de atención plena reducen la ansiedad. Practicá 10 minutos diarios para notar el cambio.",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["ansiedad", "mindfulness", "regulacion"]
        },
         {
            "pregunta_clave": "Estoy bien. Sé que mejoré mucho",
            "respuesta": "Reconocer tu propio progreso es fundamental. La sorpresa positiva refuerza la autoconfianza y el compromiso con tu bienestar.",
            "contexto_emocional_esperado": "alegria", # Cambiado de 'sorpresa' a 'alegria' para mayor coherencia
            "tags": ["autoconocimiento", "progreso", "refuerzo_positivo"]
        }
    ],

    "TutorApoyos": [
         {
            "pregunta_clave": "¿Qué apoyos puedo solicitar para continuar mis estudios?",
            "respuesta": """Podés pedir intérpretes, acompañantes pedagógicos, materiales accesibles o adaptaciones curriculares.
Cada universidad cuenta con un área de inclusión o bienestar estudiantil que gestiona estos apoyos.
Estos derechos están garantizados por la Ley Nacional 26.206 de Educación y la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["educacion", "inclusion", "accesibilidad", "ley_26206", "ley_26378"]
        },
         {
            "pregunta_clave": "No sé a quién acudir para tramitar mis beneficios",
            "respuesta": """Podés acercarte a la oficina municipal de discapacidad o a la Agencia Nacional de Discapacidad (ANDIS).
Allí te orientarán paso a paso y de forma gratuita sobre pensiones, transporte o programas de apoyo.
También podés comunicarte con el Programa ADAJUS del Ministerio de Justicia para recibir asistencia en trámites legales.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["tramites", "orientacion", "beneficios", "adajus", "andis"]
        },
        {
            "pregunta_clave": "Siento que no me escuchan cuando pido ayuda",
            "respuesta": """Es importante que tus necesidades sean reconocidas. Podés documentar tus pedidos y solicitar acompañamiento institucional.
La Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378) garantiza tu derecho a participar activamente en las decisiones que te afectan.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["derechos", "empoderamiento", "autonomia", "ley_26378"]
        },
        {
            "pregunta_clave": "¿Dónde puedo conseguir acompañamiento para buscar empleo?",
            "respuesta": """Existen programas de inserción laboral con apoyos personalizados.
La Agencia Nacional de Discapacidad (ANDIS) y la Red de Oficinas de Empleo pueden orientarte.
Recordá que la Ley Nacional 22.431 y el Decreto 312/2010 establecen un cupo del 4% en el empleo público para personas con discapacidad
y promueven la inclusión laboral en todos los sectores.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["empleo", "inclusion_laboral", "acompanamiento", "ley_22431", "cupo_laboral", "andis"]
        },
        {
            "pregunta_clave": "¿Qué significa tener apoyos para la toma de decisiones?",
            "respuesta": """Significa contar con personas, intérpretes o herramientas que te ayuden a comprender opciones y elegir lo que consideres mejor, sin que otros decidan por vos.
Este principio está reconocido por la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["autonomia", "derechos", "inclusion", "ley_26378"]
        },
        {
            "pregunta_clave": "Tengo miedo de perder mi pensión si empiezo a trabajar",
            "respuesta": """Podés mantener tu pensión siempre que tus ingresos no superen ciertos límites.
Por ejemplo, si accedés a un empleo formal, podés seguir cobrando la pensión si el salario bruto no supera el 50% del Salario Mínimo Vital y Móvil (SMVM).
Además, existen programas de compatibilización laboral gestionados por la Agencia Nacional de Discapacidad (ANDIS).
Lo importante es informar el nuevo ingreso para mantener tu situación regularizada.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["pension", "empleo", "seguridad_social", "andis", "compatibilizacion"]
        },
        {
            "pregunta_clave": "¿Qué beneficios tengo con el Certificado Único de Discapacidad?",
            "respuesta": """El Certificado Único de Discapacidad (CUD) te permite acceder a varios beneficios sin costo:
transporte público gratuito en todo el país, cobertura del 100% en medicamentos y tratamientos vinculados a tu discapacidad,
exención de algunos impuestos automotores y prioridad en programas de empleo y educación inclusiva.
Por ejemplo, podés viajar con un acompañante de forma gratuita o solicitar becas de formación adaptadas.
Es un documento clave para garantizar tus derechos y promover tu autonomía.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["cud", "beneficios", "salud", "transporte", "educacion_inclusiva"]
        },
        {
            "pregunta_clave": "¿Cómo puedo reclamar si me niegan un apoyo o beneficio?",
            "respuesta": """Tenés derecho a recibir una respuesta escrita y fundada.
Si no te la entregan, podés presentar un reclamo formal ante la Defensoría del Pueblo o el Ministerio de Justicia y Derechos Humanos.
También podés pedir asesoramiento gratuito en el Programa ADAJUS para iniciar la gestión correspondiente.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["reclamos", "defensoria", "derechos", "adajus"]
        },
        {
            "pregunta_clave": "¿Qué apoyos puedo tener para acceder a un empleo?",
            "respuesta": """Podés solicitar ajustes razonables como horarios flexibles, acompañamiento laboral o tecnología asistiva.
Estos derechos están garantizados por la Ley 22.431 y por la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).
El empleador está obligado a implementarlos sin costo para vos.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleo", "ajustes_razonables", "inclusion", "ley_22431", "ley_26378"]
        },
        {
            "pregunta_clave": "Me cuesta entender la información de los trámites",
            "respuesta": """Es tu derecho recibir información clara, completa y en formatos accesibles.
La Ley 26.653 sobre Accesibilidad de la Información, la Ley 27.275 de Acceso a la Información Pública
y la Ley 22.431 establecen que los organismos públicos deben garantizar comprensión y brindar apoyos visuales o adaptaciones cuando sea necesario.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["accesibilidad", "informacion_publica", "tramites", "claridad", "derechos", "ley_26653", "ley_27275", "ley_22431"]
        },
        {
            "pregunta_clave": "¿Qué derechos tengo si estudio en una universidad pública?",
            "respuesta": """Las universidades deben garantizar accesibilidad, intérpretes y materiales adaptados para estudiantes con discapacidad.
Podés solicitar estos apoyos en el área de inclusión universitaria o de bienestar estudiantil, según lo establece la Ley 26.206 de Educación Nacional.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["educacion_inclusiva", "universidad", "inclusion", "ley_26206"]
        },
        {
            "pregunta_clave": "Me enoja que las oficinas públicas no sean accesibles",
            "respuesta": """Tenés derecho a exigir accesibilidad física y digital.
La Ley 24.314 y la Ley 26.653 obligan al Estado a garantizar condiciones accesibles en edificios y portales web.
Podés presentar un reclamo en la Defensoría del Pueblo o el Ministerio de Obras Públicas.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["accesibilidad", "inclusion", "reclamos", "ley_24314", "ley_26653"]
        },
        {
            "pregunta_clave": "¿Puedo viajar gratis con acompañante?",
            "respuesta": """Sí. El beneficio de transporte gratuito con CUD incluye a un acompañante cuando sea necesario.
Solo necesitás presentar tu certificado y DNI en la terminal o empresa correspondiente, según la Ley 25.635 y resoluciones de la CNRT.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["transporte", "beneficios", "accesibilidad", "ley_25635"]
        },
        {
            "pregunta_clave": "Me sorprende que existan tantos programas de apoyo",
            "respuesta": """Sí, y cada año se amplían. Conocerlos te permite aprovechar recursos disponibles y compartir información con otras personas.
ANDIS, el Ministerio de Trabajo y el Ministerio de Desarrollo Social actualizan periódicamente los programas de inclusión y capacitación.""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["programas_sociales", "informacion", "empoderamiento", "andis"]
        },
        {
            "pregunta_clave": "Quiero conocer más sobre mis derechos y apoyos",
            "respuesta": """Podés consultar el portal oficial de discapacidad en www.argentina.gob.ar/andis
o acercarte a las áreas de inclusión de tu provincia.
Informarte es tu mejor herramienta de autonomía y empoderamiento ciudadano.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["derechos", "informacion", "autonomia", "andis"]
        }
    ],

    "TutorPrimerEmpleo": [
        {
            "pregunta_clave": "No tengo experiencia laboral, ¿cómo puedo empezar?",
            "respuesta": """Podés participar del Programa Jóvenes con Más y Mejor Trabajo del Ministerio de Trabajo, que te permite capacitarte y realizar prácticas en empresas.
También podés sumar experiencias en voluntariados o proyectos comunitarios, que se valoran en tu primer CV.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["primer_empleo", "capacitacion", "ministerio_trabajo"]
        },
        {
            "pregunta_clave": "¿Existe algún programa de apoyo para mi primera búsqueda laboral?",
            "respuesta": """Sí. El Programa de Inserción Laboral (PIL) ofrece incentivos económicos a empresas que contraten a personas sin experiencia o con discapacidad.
Está regulado por la Resolución 708/2010 del Ministerio de Trabajo y la Ley 22.431, que promueven la inclusión laboral en condiciones de igualdad.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["inclusion_laboral", "pila", "primer_empleo", "resolucion_708_2010", "ley_22431"]
        },
        {
            "pregunta_clave": "¿Dónde puedo hacer cursos gratuitos para conseguir trabajo?",
            "respuesta": """Podés inscribirte en los cursos del Portal Empleo del Ministerio de Trabajo (www.portalempleo.gob.ar) o en el Programa Fomentar Empleo, que ofrece formación profesional gratuita.
También hay capacitaciones accesibles en el INET y en la Agencia Nacional de Discapacidad (ANDIS).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["formacion", "capacitacion", "portal_empleo", "inet", "andis"]
        },
        {
            "pregunta_clave": "¿Qué derechos tengo si hago una práctica laboral?",
            "respuesta": """Las prácticas deben ser formativas y respetar tus derechos básicos: cobertura médica, seguridad e higiene, y un acompañamiento pedagógico.
Están reguladas por la Resolución 905/2010 del Ministerio de Trabajo.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["practicas", "derechos", "ministerio_trabajo", "resolucion_905_2010"]
        },
        {
            "pregunta_clave": "¿Puedo tener un empleo y seguir estudiando?",
            "respuesta": """Sí. La Ley de Contrato de Trabajo (N.º 20.744) garantiza horarios compatibles para estudiantes y la posibilidad de solicitar licencias para rendir exámenes.
Podés coordinarlo con tu empleador o a través de programas de inserción laboral joven.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["educacion", "trabajo_estudio", "derechos", "ley_20744"]
        },
        {
            "pregunta_clave": "Tengo miedo de que no me contraten por mi discapacidad",
            "respuesta": """Tenés derecho a igualdad de oportunidades laborales. La Ley 22.431 y el Decreto 312/2010 establecen el 4% de cupo en el sector público y promueven la inclusión en el sector privado.
Podés registrarte en la Red de Servicios de Empleo de ANDIS para acceder a búsquedas inclusivas.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["discapacidad", "inclusion_laboral", "ley_22431", "cupo_laboral", "andis"]
        },
        {
            "pregunta_clave": "¿Cómo puedo armar mi primer CV?",
            "respuesta": """Usá un formato claro: datos personales, formación, cursos y habilidades.
Podés incluir experiencias informales, voluntariados o proyectos personales.
El Portal Empleo ofrece plantillas gratuitas y guías paso a paso.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cv", "empleabilidad", "primer_empleo", "portal_empleo"]
        },
        {
            "pregunta_clave": "¿Qué hago si me piden trabajar sin registrarme?",
            "respuesta": """El trabajo no registrado es ilegal. Tenés derecho a exigir tu alta en AFIP y tus aportes.
Podés denunciar de forma confidencial en la línea 0800-666-4100 del Ministerio de Trabajo o a través del sitio www.argentina.gob.ar/trabajo.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["trabajo_informal", "derechos_laborales", "denuncia", "ministerio_trabajo"]
        },
        {
            "pregunta_clave": "¿Qué beneficios hay para las empresas que contratan por primera vez?",
            "respuesta": """El Programa de Inserción Laboral (Resolución 708/2010) otorga reducciones de contribuciones y apoyos económicos a empleadores que contraten jóvenes o personas con discapacidad.
Esto impulsa tu contratación en condiciones formales.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["empleadores", "beneficios", "incentivos", "pila", "resolucion_708_2010"]
        },
        {
            "pregunta_clave": "¿Dónde puedo recibir orientación para entrevistas?",
            "respuesta": """Podés practicar entrevistas en las Oficinas de Empleo municipales o en los talleres del Programa Fomentar Empleo.
También hay simuladores en línea en el Portal Empleo para practicar respuestas y lenguaje corporal.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["entrevista", "habilidades_blandas", "formacion", "portal_empleo"]
        },
        {
            "pregunta_clave": "¿Cómo puedo acceder a becas de formación o programas de entrenamiento?",
            "respuesta": """Podés inscribirte en las becas Progresar Trabajo (Resolución 905/2021) o en cursos del INET.
Están orientadas a la formación técnica y oficios, priorizando la inclusión de jóvenes y personas con discapacidad.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["progresar", "becas", "formacion_profesional", "resolucion_905_2021", "inet"]
        },
        {
            "pregunta_clave": "¿Qué puedo hacer si me rechazan en muchos trabajos?",
            "respuesta": """Es normal sentir frustración. Aprovechá el acompañamiento de los orientadores laborales del Ministerio de Trabajo o de ANDIS para revisar tu perfil y mejorar la búsqueda.
Cada intento te acerca a tu objetivo.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["orientacion_laboral", "motivacion", "resiliencia", "ministerio_trabajo", "andis"]
        },
        {
            "pregunta_clave": "¿Qué hago si me discriminan en una entrevista?",
            "respuesta": """La Ley 23.592 prohíbe la discriminación laboral por motivos de discapacidad. Podés presentar una denuncia ante el Ministerio Público de la Defensa.
O pedir acompañamiento legal gratuito en el Programa ADAJUS, que garantiza accesibilidad jurídica.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["discriminacion", "inclusion", "ministerio_justicia", "adajus", "ley_23592"]
        },
        {
            "pregunta_clave": "¿Dónde puedo consultar mis derechos laborales?",
            "respuesta": """Podés ingresar al sitio oficial del Ministerio de Trabajo (www.argentina.gob.ar/trabajo) y revisar la Guía de Derechos Laborales.
También podés pedir asesoramiento gratuito en las Oficinas de Empleo.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["derechos_laborales", "ministerio_trabajo", "asesoramiento"]
        },
        {
            "pregunta_clave": "Me sorprende que haya programas gratuitos de formación",
            "respuesta": """Sí, el Estado impulsa la formación gratuita a través del Programa Fomentar Empleo y el Portal Empleo.
Estos cursos te preparan para tu primer trabajo y están adaptados a distintas capacidades y niveles educativos.""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["formacion", "educacion", "inclusion", "portal_empleo"]
        }
    ],

    "GestorCUD": [
        {
            "pregunta_clave": "¿Qué es el Certificado Único de Discapacidad?",
            "respuesta": """El Certificado Único de Discapacidad (CUD) es un documento oficial que acredita tu discapacidad y te permite acceder a derechos y beneficios en salud, transporte, educación y trabajo.
Está regulado por la Ley 22.431 y la Convención sobre los Derechos de las Personas con Discapacidad (Ley 26.378).""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["cud", "definicion", "derechos", "ley_22431", "ley_26378"]
        },
        {
            "pregunta_clave": "¿Dónde se tramita el CUD?",
            "respuesta": """El trámite del CUD es gratuito y se realiza en las juntas evaluadoras de tu provincia o municipio.
Podés consultar las direcciones actualizadas en www.argentina.gob.ar/andis/cud.
Si necesitás apoyo para trasladarte, podés solicitar asistencia o turno prioritario.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["tramite", "andis", "accesibilidad"]
        },
        {
            "pregunta_clave": "¿Qué documentación necesito para iniciar el trámite del CUD?",
            "respuesta": """Tenés que presentar tu DNI, un resumen médico actualizado y estudios que certifiquen la condición.
Si sos menor de edad o estás a cargo de otra persona, también se solicita DNI del representante legal.
Los formularios se descargan desde el sitio oficial de ANDIS.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["documentacion", "tramites", "andis"]
        },
        {
            "pregunta_clave": "Tengo miedo de que el trámite sea complicado",
            "respuesta": """Es normal sentir eso. El proceso puede parecer largo, pero hay acompañamiento.
Las juntas evaluadoras te orientan paso a paso, y podés pedir que te expliquen todo en lenguaje claro o con apoyos visuales, según lo establece la Ley 26.653 de Accesibilidad a la Información.""",
            "contexto_emocional_esperado": "miedo",
            "tags": ["acompanamiento", "accesibilidad", "tramites", "ley_26653"]
        },
        {
            "pregunta_clave": "¿El trámite del CUD tiene costo?",
            "respuesta": """No. El trámite del CUD es totalmente gratuito en todo el país.
Si alguien te solicita un pago o arancel, hacé la denuncia ante la Agencia Nacional de Discapacidad (ANDIS).""",
            "contexto_emocional_esperado": "ira",
            "tags": ["tramite", "denuncia", "andis"]
        },
        {
            "pregunta_clave": "¿Qué beneficios obtengo al tener el CUD?",
            "respuesta": """Con el CUD accedés a transporte gratuito, cobertura del 100% en medicamentos y tratamientos vinculados a tu discapacidad,
exención de impuestos automotores, prioridad en programas de empleo inclusivo y becas educativas adaptadas.
También podés tramitar el pase libre en transporte público (Ley 25.635).""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["beneficios", "transporte", "salud", "educacion", "ley_25635"]
        },
        {
            "pregunta_clave": "¿Cuánto tiempo demora el trámite?",
            "respuesta": """Depende de la provincia adonde residís, pero suele demorar entre 30 y 90 días desde la presentación completa de la documentación.
Podés consultar el estado del trámite en la junta evaluadora o en ANDIS.""",
            "contexto_emocional_esperado": "anticipacion",
            "tags": ["plazos", "tramites", "andis"]
        },
        {
            "pregunta_clave": "¿Cuándo tengo que renovar el CUD?",
            "respuesta": """El CUD tiene una validez variable según la condición de la persona.
Generalmente se renueva cada 5 años, aunque en algunos casos puede tener vigencia permanente.
Podés verificar la fecha de vencimiento en el certificado o consultar en tu junta evaluadora.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["renovacion", "plazos", "certificados"]
        },
        {
            "pregunta_clave": "¿Puedo tramitar el CUD si no tengo obra social?",
            "respuesta": """Sí. El CUD es un derecho universal y no depende de tener obra social.
Te permite acceder a prestaciones médicas y programas de salud pública sin costo adicional.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["salud_publica", "derechos", "acceso"]
        },
        {
            "pregunta_clave": "Me rechazaron el CUD, ¿qué puedo hacer?",
            "respuesta": """Podés pedir una revisión del dictamen dentro de los 10 días hábiles posteriores a la notificación.
La ANDIS dispone de un formulario para apelaciones.
También podés solicitar acompañamiento en el Programa ADAJUS del Ministerio de Justicia.""",
            "contexto_emocional_esperado": "ira",
            "tags": ["reclamo", "andis", "derechos", "adajus"]
        },
        {
            "pregunta_clave": "¿Puedo tramitar el CUD si vivo en una zona rural?",
            "respuesta": """Sí. En muchas provincias hay operativos móviles de ANDIS que visitan localidades rurales para realizar evaluaciones.
También podés solicitar turnos priorizados si tenés dificultades de movilidad.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["zona_rural", "movilidad", "andis"]
        },
        {
            "pregunta_clave": "¿Qué hago si perdí mi CUD?",
            "respuesta": """Podés pedir un duplicado en la misma junta evaluadora donde lo tramitaste.
Solo necesitás tu DNI y una constancia policial de extravío.
El duplicado conserva la misma validez que el original.""",
            "contexto_emocional_esperado": "tristeza",
            "tags": ["duplicado", "tramite", "certificado"]
        },
        {
            "pregunta_clave": "¿Puedo usar mi CUD en todo el país?",
            "respuesta": """Sí. El CUD tiene validez nacional.
Esto significa que podés usarlo en cualquier provincia o municipio para acceder a los mismos beneficios, sin necesidad de tramitarlo nuevamente.""",
            "contexto_emocional_esperado": "alegria",
            "tags": ["cud", "validez_nacional", "beneficios"]
        },
        {
            "pregunta_clave": "¿Qué pasa si cambian mis condiciones de salud?",
            "respuesta": """Si tu condición cambia, podés solicitar una nueva evaluación médica en tu junta evaluadora.
El CUD puede actualizarse para reflejar tu situación actual y garantizar que sigas recibiendo los apoyos adecuados.""",
            "contexto_emocional_esperado": "confianza",
            "tags": ["evaluacion", "actualizacion", "salud"]
        },
        {
            "pregunta_clave": "Me sorprende que el CUD tenga tantos beneficios",
            "respuesta": """Sí, es una herramienta fundamental para el ejercicio de tus derechos.
Además de beneficios sociales y de salud, te facilita la inclusión educativa y laboral.
Podés informarte más en el sitio oficial de ANDIS (www.argentina.gob.ar/andis/cud).""",
            "contexto_emocional_esperado": "sorpresa",
            "tags": ["derechos", "beneficios", "andis"]
        }
    ],
}

if __name__ == "__main__":
    # Pequeño script de validación al ejecutar directamente
    print(f"--- Validación de src/expert_kb.py ---")
    print(f"Tutores cargados: {len(EXPERT_KB)} -> {list(EXPERT_KB.keys())}")
    total_intenciones = sum(len(v) for v in EXPERT_KB.values())
    print(f"Total de intenciones definidas: {total_intenciones}")

    # Chequeo de emociones válidas
    valid_emotions = {"Alegria", "Tristeza", "Ira", "Miedo", "Sorpresa", "Confianza", "Anticipacion", "Neutral"} # Añadido Neutral para defaults
    errors = []
    for tutor, intentions in EXPERT_KB.items():
        for i, intention in enumerate(intentions):
            emo = intention.get("contexto_emocional_esperado", "").capitalize()
            if emo not in valid_emotions:
                errors.append(f"Error en {tutor}[{i}]: Emoción '{intention.get('contexto_emocional_esperado')}' inválida. Clave: '{intention.get('pregunta_clave')}'")

    if not errors:
        print("✅ Validación de emociones esperadas: OK (Todas pertenecen a las 7 base + Neutral)")
    else:
        print("\n❌ ERRORES DE VALIDACIÓN ENCONTRADOS:")
        for error in errors:
            print(error)

    print("-" * 40)

