from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from .vectorstore import get_qa_chain

# Initialize chain once (only when the server starts)
qa_chain = get_qa_chain()

def home(request):
    return render(request, 'index.html')


@csrf_exempt
def ask_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()
            q_type = data.get("type", "").lower()
            q_level = data.get("level", "").lower()

            if not question:
                return JsonResponse({"error": "Missing question"}, status=400)

            # Construct prompt dynamically based on dropdown
            if q_type == "lab practical":
                prompt = (
                    f"You are a Unix/Linux lab instructor. Based on the experiment: '{question}', "
                    f"generate 5 lab practical coding questions of {q_level} difficulty. "
                    f"Each should involve scripting, file handling, control structures (if, loops), etc. "
                    f"Format them like this:\n\n**Questions**\n1. ...\n2. ...\n\n**Answers**\n1. ...\n2. ..."
                )
            else:
                prompt = (
                    f"You are an expert in Unix/Linux viva preparation. Based on this topic: '{question}', "
                    f"generate 5 {q_level} level viva questions with short descriptive answers. "
                    f"Format them like this:\n\n**Questions**\n1. ...\n2. ...\n\n**Answers**\n1. ...\n2. ..."
                )

            # Debug prompt (optional)
            print("\n--- Prompt Sent to LLM ---\n", prompt)

            response = qa_chain.invoke({"query": prompt})
            return JsonResponse({"answer": response["result"]})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)



