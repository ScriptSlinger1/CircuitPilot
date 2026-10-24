from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView
from openai import OpenAI
# Create your views here.

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=45.0,
    max_retries=2,
)

class LandingPageView(TemplateView):
    template_name = 'cpai/landing.html'


def main_func_temp(request):
    question = None
    response = None
    if request.method == 'POST':
        question = request.POST.get('question')
        prompt = f'You are circuit pilot ai that only helps with engineering questions and how to build devices here is the users question: {question}'
        ai_res = client.responses.create(model="gpt-5.6-luna", input=prompt)
        response = ai_res.output_text
    return render(request, 'cpai/main.html', {'question': question, 'response': response})