# testing/test_queries.py

from services.document_processor import DocumentProcessor
from services.retriever import Retriever
from services.generator import Generator

print("📄 Loading PDF and creating chunks for testing...")
dp = DocumentProcessor()
pages = dp.extract_pages("data/Boeing_B737_Manual.pdf")
chunks = dp.create_chunks(pages)

retriever = Retriever(chunks)
generator = Generator()
print("✅ RAG system ready for testing!\n")


def test_query(question: str):
    print(f"\n❓ Question: {question}\n")

    # ------------------------------
    # Step 1: Retrieve chunks
    # ------------------------------
    retrieved_chunks = retriever.retrieve(question, top_k=5)

    if not retrieved_chunks:
        print("⚠️ No relevant chunks found.")
        return

    # ------------------------------
    # Step 2: Generate LLM Answer
    # ------------------------------
    try:
        answer, _ = generator.generate_answer(question, retrieved_chunks)
        answer = answer.strip() if answer else "(No answer returned)"
    except Exception as e:
        print("❌ Error during LLM inference:", e)
        return

    # ------------------------------
    # Step 3: Collect referenced pages
    # ------------------------------
    try:
        referenced_pages = Retriever.collect_pages_from_retrieved_chunks(retrieved_chunks)
    except Exception:
        referenced_pages = sorted({p for c in retrieved_chunks for p in c.get("pages", [])})

    # ------------------------------
    # Step 4: Print Output
    # ------------------------------
    print("✅ Answer:\n", answer)
    print("📄 Referenced pages:", referenced_pages)
    print("📊 Retriever scores:")

    for c in retrieved_chunks:
        cid = c.get("chunk_id", "N/A")
        score = c.get("score", 0.0)
        pages = c.get("pages", [])
        print(f"  - Chunk {cid} | score: {score:.3f} | pages: {pages}")


if __name__ == "__main__":
    questions = [
        "I'm calculating our takeoff weight for a dry runway. We're at 2,000 feet pressure altitude, and the OAT is 50°C. What's the climb limit weight?",
        "We're doing a Flaps 15 takeoff. Remind me, what is the first flap selection we make during retraction, and at what speed?",
        "We're planning a Flaps 40 landing on a wet runway at a 1,000-foot pressure altitude airport. If the wind-corrected field length is 1,600 meters, what is our field limit weight?",
        "Reviewing the standard takeoff profile: After we're airborne and get a positive rate of climb, what is the first action we take?",
        "Looking at the panel scan responsibilities for when the aircraft is stationary, who is responsible for the forward aisle stand?",
        "For a standard visual pattern, what three actions must be completed prior to turning base?",
        "When the Pilot Not Flying (PNF) makes CDU entries during flight, what must the Pilot Flying (PF) do prior to execution?",
        "I see an amber 'STAIRS OPER' light illuminated on the forward attendant panel; what does that light indicate?",
        "We've just completed the engine start. What is the correct configuration for the ISOLATION VALVE switch during the After Start Procedure?",
        "During the Descent and Approach procedure, what action is taken with the AUTO BRAKE select switch, and what is the Pilot Flying's final action regarding the autobrake system during the Landing Roll procedure?"
    ]

    for q in questions:
        test_query(q)
