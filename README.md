LibAI

LibAI is an AI-powered IIMB Library Reference Assistant built with Streamlit
and the Ollama Cloud API.

This initial version provides general library and research assistance. It does
not yet retrieve answers from IIMB Library PDFs or other institutional files.

1. Create an Ollama API key

Sign in at https://ollama.com.

Open https://ollama.com/settings/keys.

Create an API key and copy it securely.

Do not save the real key anywhere in this GitHub repository.

You can list the cloud models available to your account from Windows
PowerShell:

$env:OLLAMA_API_KEY="PASTE_YOUR_KEY_HERE"

curl.exe https://ollama.com/api/tags `
  -H "Authorization: Bearer $env:OLLAMA_API_KEY"

The application defaults to gpt-oss:120b. If that model is not listed for
your account, use one of the model names returned by the command above.

2. Upload the project to GitHub

Create a repository named libAI, then upload all files and folders from this
package. Keep this exact structure:

libAI/
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
├── README.md
├── requirements.txt
└── streamlit_app.py

The example secrets file is safe to upload because it contains no real key.

3. Deploy on Streamlit Community Cloud

Open https://share.streamlit.io and sign in with GitHub.

Select Create app.

Choose your libAI repository and the main branch.

Set the main file path to streamlit_app.py.

Open Advanced settings and paste the following into Secrets:

OLLAMA_API_KEY = "YOUR_REAL_OLLAMA_API_KEY"
OLLAMA_MODEL = "gpt-oss:120b"

Replace the model if your Ollama account lists a different cloud model.

Select Deploy.

4. Test LibAI

Ask:

Explain what a library discovery service is.

If the app reports an authentication error, replace the API key in Streamlit
Secrets. If it reports that the model is unavailable, replace OLLAMA_MODEL
with a model returned by the /api/tags request.

Local testing (optional)

Create a local secrets file that will remain excluded from Git:

.streamlit/secrets.toml

Add the same two TOML entries shown above. Then run:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py

On Windows PowerShell, activate the virtual environment with:

.venv\Scripts\Activate.ps1

Security

Never commit .streamlit/secrets.toml, .env, API keys, or .pem files.

If a key is exposed, revoke it in Ollama and create a replacement.

Update Streamlit Secrets whenever the key or cloud model changes.

Next phase

To answer IIMB-specific questions reliably, add a retrieval layer containing
approved Library FAQs, policies, services, database information, and other
institutional documents. Until then, LibAI intentionally declines to invent
IIMB-specific facts.
