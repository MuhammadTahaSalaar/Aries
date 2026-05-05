import asyncio
import json
import time
from src.shared.config import get_settings
from src.nlp.ner.slm_inference import run_ner_inference_slm
from src.nlp.summarizer.slm_inference import run_summarization_inference_slm
from src.nlp.summarizer.schemas import SummarizeMode

async def main():
    settings = get_settings()
    settings.use_slm = True
    
    print("================================")
    print("Testing NER Endpoint")
    print("================================")
    ner_text = "The APT group Lazarus deployed TrickBot malware targeting Windows 10 systems at Acme Corp. The attack exploited CVE-2021-44228 (Log4Shell) to gain initial access. C2 traffic observed connecting to 185.220.101.42."
    
    start = time.time()
    try:
        ner_res = await run_ner_inference_slm(ner_text, settings)
        print(f"NER Output ({time.time() - start:.2f}s):")
        print(json.dumps(ner_res.model_dump(), indent=2))
    except Exception as e:
        print(f"NER Error: {e}")

    print("\n================================")
    print("Testing Summarizer Endpoint")
    print("================================")
    sum_text = "On March 15, 2024, the SOC detected a multi-stage attack. A spear-phishing email exploited CVE-2023-36884 to deploy malware. The attacker used PowerShell to download payloads from 203.0.113.50, established persistence via scheduled tasks, moved laterally using stolen credentials, accessed the domain controller, and exfiltrated 2.3GB of financial data. The attack was attributed to FIN7."
    
    start = time.time()
    try:
        sum_res = await run_summarization_inference_slm(sum_text, SummarizeMode.EXECUTIVE, settings)
        print(f"Summarizer Output ({time.time() - start:.2f}s):")
        print(json.dumps(sum_res.model_dump(), indent=2))
    except Exception as e:
        print(f"Summarizer Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
