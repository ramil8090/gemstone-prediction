import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

request = {
    "image_url": "https://www.gia.edu/images/polished-ruby.png"
}

result = requests.post(url, json=request).json()
print(result)