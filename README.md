
Create a virtual environment and install dependencies.

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]" --no-cache-dir
```

Build a Docker image for API

```
docker build -t handwritten-digit-string-recognition/api -f api/Dockerfile .
```

Run the Docker image

```
docker run -p 9000:8080 -it --rm handwritten-digit-string-recognition/api
```

Example command to send a request

```
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" -H 'Content-Type: application/json' -d '{ "image": "data:image/png;base64,'$(base64 -i test_image.png)'" }'
```

Run Streamlit

```
streamlit run streamlit/app.py
```
