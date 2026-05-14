import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, DCTERMS

try:
    import ollama
    client = ollama.Client(host='http://localhost:11434')
    models = client.list()
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

st.set_page_config(page_title="Knowledge Graph Navigator", layout="wide")

COMMON_LABEL_PROPS = [RDFS.label]
COMMON_DESC_PROPS = [RDFS.comment]


def get_base_dir():
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
TTL_FILE_PATH = BASE_DIR / "kg_with_instances.ttl"

@st.cache_resource
def load_graph_from_file(local_path):
    g = Graph()
    g.parse(local_path, format="turtle")
    return g


def shorten_uri(uri):
    text = str(uri)
    if "#" in text:
        return text.split("#")[-1]
    if "/" in text:
        return text.rstrip("/").split("/")[-1]
    return text


def get_label(g, node):
    for prop in COMMON_LABEL_PROPS:
        for obj in g.objects(node, prop):
            if isinstance(obj, Literal):
                return str(obj)
    if isinstance(node, URIRef):
        return shorten_uri(node)
    return str(node)


def get_comment(g, node):
    for prop in COMMON_DESC_PROPS:
        for obj in g.objects(node, prop):
            if isinstance(obj, Literal):
                return str(obj)
    return ""


def classify_node(g, node):
    types = [get_label(g, t) for t in g.objects(node, RDF.type) if t != OWL.NamedIndividual]
    return ", ".join(types) if types else "Unclassified"


def list_entity_candidates(g):
    candidates = []

    for s in set(g.subjects()):
        if isinstance(s, Literal):
            continue
        candidates.append(
            {
                "label": get_label(g, s),
                "uri": str(s),
                "type": classify_node(g, s),
            }
        )

    if not candidates:
        return pd.DataFrame(columns=["label", "uri", "type"])

    df = pd.DataFrame(candidates)
    for col in ["label", "uri", "type"]:
        if col not in df.columns:
            df[col] = ""

    return df.sort_values(["label", "type"]).reset_index(drop=True)


def search_entities(df, query):
    if df.empty:
        return df
    if not query:
        return df.head(100)

    mask = (
        df["label"].str.contains(query, case=False, na=False)
        | df["uri"].str.contains(query, case=False, na=False)
        | df["type"].str.contains(query, case=False, na=False)
    )
    return df.loc[mask].reset_index(drop=True)


def get_node_from_selection(df, selection_label):
    if df.empty or "display" not in df.columns or "uri" not in df.columns:
        return None

    row = df.loc[df["display"] == selection_label]
    if row.empty:
        return None

    uri = str(row.iloc[0]["uri"]).strip()
    if not uri:
        return None

    return URIRef(uri)


def describe_entity(g, node):
    outgoing = []
    incoming = []

    for p, o in g.predicate_objects(node):
        outgoing.append(
            {
                "predicate": get_label(g, p),
                "object": get_label(g, o),
                "object_uri": str(o) if isinstance(o, URIRef) else "",
            }
        )

    for s, p in g.subject_predicates(node):
        incoming.append(
            {
                "subject": get_label(g, s),
                "subject_uri": str(s) if isinstance(s, URIRef) else "",
                "predicate": get_label(g, p),
            }
        )

    return pd.DataFrame(outgoing), pd.DataFrame(incoming)


def is_metadata_predicate(predicate):
    metadata_predicates = {
        str(RDFS.label),
        str(RDFS.comment),
        str(RDF.type),
        "http://purl.org/dc/terms/source",
        "http://purl.org/dc/elements/1.1/source",
    }
    return str(predicate) in metadata_predicates


def build_ego_network(g, center_node, max_edges=60):
    G = nx.DiGraph()
    center_label = get_label(g, center_node)
    G.add_node(str(center_node), label=center_label, kind="center")

    edge_count = 0

    for p, o in g.predicate_objects(center_node):
        if edge_count >= max_edges:
            break

        # Hide metadata from visual graph
        if is_metadata_predicate(p):
            continue

        obj_key = str(o)
        G.add_node(obj_key, label=get_label(g, o), kind="out")
        G.add_edge(str(center_node), obj_key, label=get_label(g, p))
        edge_count += 1

    for s, p in g.subject_predicates(center_node):
        if edge_count >= max_edges:
            break

        # Hide metadata from visual graph
        if is_metadata_predicate(p):
            continue

        sub_key = str(s)
        G.add_node(sub_key, label=get_label(g, s), kind="in")
        G.add_edge(sub_key, str(center_node), label=get_label(g, p))
        edge_count += 1

    return G


def draw_interactive_pyvis(graph_nx, height="700px", width="100%"):
    if not PYVIS_AVAILABLE:
        st.info("Pyvis is not installed. Run: pip install pyvis")
        return

    with st.expander("Graph layout settings", expanded=False):
        physics_enabled = st.toggle("Enabled", value=True)
        solver = st.selectbox(
            "Layout algorithm",
            ["barnesHut", "forceAtlas2Based", "repulsion", "hierarchicalRepulsion"],
            index=0
        )

    net = Network(height=height, width=width, directed=True)

    for node, data in graph_nx.nodes(data=True):
        label = data.get("label", str(node))
        net.add_node(str(node), label=label, title=label)

    for source, target, data in graph_nx.edges(data=True):
        edge_label = data.get("label", "")
        net.add_edge(str(source), str(target), label=edge_label, title=edge_label)

    if solver == "barnesHut":
        net.barnes_hut()
    elif solver == "forceAtlas2Based":
        net.force_atlas_2based()
    elif solver == "repulsion":
        net.repulsion()
    elif solver == "hierarchicalRepulsion":
        net.hrepulsion()

    net.toggle_physics(physics_enabled)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html = Path(tmp_file.name).read_text(encoding="utf-8")

    components.html(html, height=750, scrolling=True)

def normalise_query_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def score_entity_match(query, row):
    query_norm = normalise_query_text(query)
    label_norm = normalise_query_text(row.get("label", ""))
    type_norm = normalise_query_text(row.get("type", ""))
    uri_norm = normalise_query_text(row.get("uri", ""))

    if not query_norm:
        return 0

    score = 0
    q_tokens = set(query_norm.split())
    label_tokens = set(label_norm.split())
    type_tokens = set(type_norm.split())
    uri_tokens = set(uri_norm.split())

    score += len(q_tokens & label_tokens) * 4
    score += len(q_tokens & type_tokens) * 2
    score += len(q_tokens & uri_tokens)

    if query_norm in label_norm:
        score += 10
    if query_norm in uri_norm:
        score += 4

    return score


def retrieve_relevant_entities(entities_df, question, top_k=5):
    if entities_df.empty:
        return pd.DataFrame(columns=["label", "uri", "type", "match_score"])

    working = entities_df.copy()
    working["match_score"] = working.apply(lambda r: score_entity_match(question, r), axis=1)
    working = working.loc[working["match_score"] > 0].sort_values("match_score", ascending=False)
    return working.head(top_k).reset_index(drop=True)


def get_local_triples_for_node(g, node, max_outgoing=10, max_incoming=10):
    triples = []

    count = 0
    for p, o in g.predicate_objects(node):
        if count >= max_outgoing:
            break
        triples.append((get_label(g, node), get_label(g, p), get_label(g, o)))
        count += 1

    count = 0
    for s, p in g.subject_predicates(node):
        if count >= max_incoming:
            break
        triples.append((get_label(g, s), get_label(g, p), get_label(g, node)))
        count += 1

    return triples


def build_qa_context(g, matched_entities_df, max_entities=3):
    if matched_entities_df.empty:
        return "No relevant entities were retrieved from the knowledge graph.", []

    context_lines = []
    evidence_rows = []

    for _, row in matched_entities_df.head(max_entities).iterrows():
        node = URIRef(row["uri"])
        label = row["label"]
        node_type = row["type"]
        context_lines.append(f"Entity: {label} | Type: {node_type}")

        triples = get_local_triples_for_node(g, node)
        for s, p, o in triples:
            context_lines.append(f"- {s} | {p} | {o}")
            evidence_rows.append(
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "seed_entity": label,
                }
            )
        context_lines.append("")

    return "\n".join(context_lines).strip(), evidence_rows

def build_evidence_graph(evidence_rows):
    G = nx.DiGraph()

    for row in evidence_rows:
        s = str(row["subject"])
        p = str(row["predicate"])
        o = str(row["object"])

        G.add_node(s, label=s)
        G.add_node(o, label=o)
        G.add_edge(s, o, label=p)

    return G


TOPIC_TO_CLASSES = {
    "programme": ["Programme", "School", "College", "ContactPoint"],
    "course": ["Course", "Programme", "School", "Staff", "AcademicYear"],
    "staff": ["Staff", "Person", "School", "ResearchGroup"],
    "research": ["ResearchProject", "ResearchGroup", "ResearchCentre", "Topic", "Staff"],
    "policy": ["Policy", "Regulation", "Document"],
    "scholarship": ["Scholarship", "Programme", "School"],
    "contact": ["ContactPoint", "Staff", "School", "Programme"],
    "event": ["Event", "School", "Staff"],
    "general": []
}

def classify_question_topic(question, model_name="llama3.1:8b"):
    """
    Use local LLM only for lightweight intent/topic classification.
    """

    if not OLLAMA_AVAILABLE:
        return "general"

    prompt = f"""
You are classifying questions for a university knowledge graph.

Choose ONLY ONE topic from this list:

programme
course
staff
research
policy
scholarship
contact
event
general

Return ONLY the topic word.

Question:
{question}
"""

    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt
        )

        if isinstance(response, dict):
            topic = response.get("response", "").strip().lower()
        else:
            topic = str(response).strip().lower()

        if topic in TOPIC_TO_CLASSES:
            return topic

        return "general"

    except Exception:
        return "general"


def ask_local_llm(question, context, model_name="llama3.1:8b"):
    """
    Ask local LLM using filtered KG context only.
    """

    if not OLLAMA_AVAILABLE:
        return None, "Ollama Python package is not installed. Run: pip install ollama"

    # Step 1 — detect topic
    topic = classify_question_topic(question, model_name)

    # Step 2 — build stricter prompt
    prompt = f"""
You are assisting users with a university knowledge graph.

Question topic:
{topic}

The context below contains RDF-style triples:

Subject | Predicate | Object

Rules:
- Use ONLY the provided triples.
- Do NOT use external knowledge.
- Do NOT invent facts.
- If the answer is not clearly supported, say:
"The knowledge graph does not contain enough information to answer this question."

Instructions:
- Focus only on entities relevant to the topic.
- Use predicates to connect information.
- Prefer short and factual answers.
- Mention entity labels exactly as written when possible.
- Ignore unrelated triples.

Retrieved triples:
{context}

User question:
{question}
"""

    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt
        )

        if isinstance(response, dict):
            return response.get("response", ""), None

        return str(response), None

    except Exception as e:
        return None, str(e)


st.title("Knowledge Graph Navigator")
st.caption("Streamlit interface for exploring an RDF/Turtle knowledge graph")

with st.sidebar:
    st.header("Knowledge graph")
    st.write("Using local Turtle file:")
    st.code(str(TTL_FILE_PATH))
    st.write(f"TTL exists: {TTL_FILE_PATH.exists()}")
    st.markdown("---")
    st.header("Packages")
    st.write(f"Ollama available: {OLLAMA_AVAILABLE}")
    st.write(f"Pyvis available: {PYVIS_AVAILABLE}")

try:
    graph = load_graph_from_file(local_path=TTL_FILE_PATH)

except Exception as e:
    st.error(f"Failed to load graph: {e}")

    st.markdown("### Raw TTL preview")

    try:
        with open(TTL_FILE_PATH, "r", encoding="utf-8") as f:
            ttl_text = f.read()

        st.text_area(
            "TTL file content",
            ttl_text,
            height=500
        )

    except Exception as file_error:
        st.error(f"Could not read TTL file: {file_error}")

    st.stop()
    
st.success(f"Graph loaded successfully. Total triples: {len(graph):,}")

entities_df = list_entity_candidates(graph)
if entities_df.empty:
    st.error("No entities were found in the knowledge graph. Check the TTL file and parsing output.")
    st.stop()

main_tab1, main_tab2, main_tab3 = st.tabs([
    "Ask the graph",
    "SPARQL query",
    "Browse entities"
])

with main_tab1:
    st.subheader("Ask a question")

    # model selection
    model_name = st.text_input("Ollama model name", value="llama3.1:8b")

    # user question
    user_question = st.text_input("Ask a question about the university knowledge graph")

    # ensure session state exists
    if "evidence_rows" not in st.session_state:
        st.session_state["evidence_rows"] = []

    if st.button("Answer question"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            # 1. classify question topic
            topic = classify_question_topic(
                user_question,
                model_name=model_name
            )

            allowed_classes = TOPIC_TO_CLASSES.get(topic, [])

            st.markdown("### Detected topic")
            st.write(topic)

            # 2. filter entities by topic/classes
            if allowed_classes:
                filtered_entities_df = entities_df[
                    entities_df["type"].isin(allowed_classes)
                ].copy()
            else:
                filtered_entities_df = entities_df.copy()

            # 3. retrieve relevant entities from filtered set only
            matched_entities = retrieve_relevant_entities(
                filtered_entities_df,
                user_question,
                top_k=5
            )

            # 4. build context + evidence
            context_text, evidence_rows = build_qa_context(
                graph,
                matched_entities,
                max_entities=3
            )

            st.session_state["evidence_rows"] = evidence_rows

            # ---- matched entities ----
            st.markdown("### Matched entities")

            if matched_entities.empty:
                st.info("No relevant entities were retrieved from the graph.")
            else:
                st.dataframe(
                    matched_entities[["label", "type", "uri", "match_score"]],
                    width="stretch",
                    height=220,
                )

            # ---- LLM answer ----
            answer, error = ask_local_llm(
                user_question,
                context_text,
                model_name=model_name
            )

            st.markdown("### Answer")
            if error:
                st.error(error)
            elif answer:
                st.write(answer)
            else:
                st.info("No answer was returned.")

            # ---- evidence triples ----
            st.markdown("### Evidence triples")
            evidence_df = pd.DataFrame(evidence_rows)

            if evidence_df.empty:
                st.info("No evidence triples available.")
            else:
                st.dataframe(evidence_df, width="stretch", height=250)

            # ---- evidence knowledge graph ----
            st.markdown("### Evidence knowledge graph")

            if evidence_rows:
                evidence_graph = build_evidence_graph(evidence_rows)

                if evidence_graph.number_of_edges() == 0:
                    st.info("No graph structure available for the retrieved evidence.")
                else:
                    draw_interactive_pyvis(evidence_graph)
            else:
                st.info("No evidence graph available.")

with main_tab2:
    st.subheader("SPARQL query")

    example_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o .
}
LIMIT 20"""

    sparql_text = st.text_area(
        "Enter a SPARQL query",
        value=example_query,
        height=180
    )

    if st.button("Run SPARQL"):
        sparql_df, sparql_error = run_sparql(graph, sparql_text)

        if sparql_error:
            st.error(f"SPARQL error: {sparql_error}")
        elif sparql_df is None or sparql_df.empty:
            st.info("Query returned no results.")
        else:
            st.dataframe(sparql_df, use_container_width=True)
            
with main_tab3:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Entity search")
        query = st.text_input("Search by label, URI, or type", key="browse_query")
        results_df = search_entities(entities_df, query)

        if results_df.empty:
            st.warning("No matching entities found.")
            st.stop()

        results_df = results_df.copy()
        results_df["label"] = results_df["label"].fillna("").astype(str)
        results_df["type"] = results_df["type"].fillna("Unclassified").astype(str)
        results_df["uri"] = results_df["uri"].fillna("").astype(str)
        results_df["display"] = (
            results_df["label"]
            + "  |  "
            + results_df["type"]
            + "  |  "
            + results_df["uri"].map(shorten_uri)
        )

        selection = st.selectbox(
            "Select an entity",
            results_df["display"].tolist(),
            key="browse_selection"
        )

        selected_node = get_node_from_selection(results_df, selection)

        if selected_node is None:
            st.warning("Could not resolve the selected entity.")
            st.stop()

        st.markdown("### Matching entities")
        st.dataframe(
            results_df[["label", "type", "uri"]],
            width="stretch",
            height=300,
        )

    with col2:
        st.subheader("Entity details")
        entity_label = get_label(graph, selected_node)
        entity_type = classify_node(graph, selected_node)
        entity_comment = get_comment(graph, selected_node)

        st.markdown(f"**Label:** {entity_label}")
        st.markdown(f"**URI:** `{selected_node}`")
        st.markdown(f"**Type:** {entity_type}")

        # original source webpages
        sources = sorted(set(
            str(s) for s in graph.objects(selected_node, DCTERMS.source)
            if str(s).startswith("http")
        ))
        st.markdown("**Original source links:**")
    
        if sources:
            for url in sources:
                st.markdown(f"- [{url}]({url})")
        else:
            st.info("No source links found.")    
            
        if entity_comment:
            st.markdown(f"**Description:** {entity_comment}")

        outgoing_df, incoming_df = describe_entity(graph, selected_node)

        tab1, tab2, tab3 = st.tabs(
            ["Outgoing relations", "Incoming relations", "Neighbourhood graph"]
        )

        with tab1:
            if outgoing_df.empty:
                st.info("No outgoing relations available.")
            else:
                st.dataframe(
                    outgoing_df,
                    width="stretch",
                    height=350,
                )

        with tab2:
            if incoming_df.empty:
                st.info("No incoming relations available.")
            else:
                st.dataframe(
                    incoming_df,
                    width="stretch",
                    height=350,
                )

        with tab3:
            ego = build_ego_network(graph, selected_node)

            if ego.number_of_edges() == 0:
                st.info("No graph neighbourhood available for this node.")
            else:
                draw_interactive_pyvis(ego)
