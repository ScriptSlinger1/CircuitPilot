from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

# Create your views here.

def test_view() -> HttpResponse:
    return HttpResponse('test')