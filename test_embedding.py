import os
from dotenv import load_dotenv
import voyageai

load_dotenv()

client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

result = client.embed(["Law 36: Leg Before Wicket"], model="voyage-4-lite")

vector = result.embeddings[0]
print(f"Total vector length: {len(vector)}")
print(vector[:5])
