from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import json

from .vectorstore import get_qa_chain

# Load the QA chain once
qa_chain = get_qa_chain()

# Homepage view
def home(request):
    return render(request, 'index.html')  # Make sure templates/index.html exists

# API endpoint to handle chatbot questions
@csrf_exempt
def ask_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get("question", "")
            if not query:
                return JsonResponse({"error": "Missing question"}, status=400)
            print(f"[INFO] Received question: {query}")
            response = qa_chain.invoke({"query": query})
            print(f"[INFO] Raw response from LLM: {response}")
            return JsonResponse({"answer": response["result"]})
            if not response or "result" not in response:
                return JsonResponse({"error": "No answer returned from model"}, status=500)

            return JsonResponse({"answer": response["result"]})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
