FINTECH_TOPICS = [
    {
        "domain": "CKYC (Central KYC)",
        "subtopics": [
            "What is CKYC and how is it different from regular KYC?",
            "How is CKYC processed step-by-step?",
            "What documents are required for CKYC registration?",
            "What is the role of NSDL/CKYC in the Indian financial system?",
            "How does CKYC benefit customers and financial institutions?",
            "What is the CKYC number and how is it used across institutions?",
            "How to update CKYC details?",
            "What is the difference between CKYC and e-KYC?",
        ]
    },
    {
        "domain": "KYC (Know Your Customer)",
        "subtopics": [
            "What is KYC and why is it mandatory in India?",
            "What are the different types of KYC (Aadhaar-based, In-Person, Video)?",
            "What documents are accepted for KYC verification?",
            "What is the regulatory framework for KYC under RBI and SEBI?",
            "What happens if KYC is not done or is expired?",
            "How often should KYC be updated?",
            "What is periodic KYC vs initial KYC?",
        ]
    },
    {
        "domain": "AML (Anti-Money Laundering)",
        "subtopics": [
            "What is Anti-Money Laundering (AML) and why is it important?",
            "What are the stages of money laundering: Placement, Layering, Integration?",
            "What is the Prevention of Money Laundering Act (PMLA) 2002?",
            "What are the obligations of financial institutions under AML?",
            "What is Suspicious Transaction Reporting (STR) and how is it filed?",
            "What is Cash Transaction Report (CTR) and when is it required?",
            "What is the role of FIU-IND in AML compliance?",
            "How to implement an AML compliance program for a fintech?",
            "What is KYC's role in AML compliance?",
        ]
    },
    {
        "domain": "e-KYC / Aadhaar KYC",
        "subtopics": [
            "What is e-KYC and how does it work with Aadhaar?",
            "What is the difference between e-KYC and physical KYC?",
            "What is Aadhaar Paperless Offline e-KYC?",
            "How does UIDAI facilitate e-KYC?",
            "What are the consent requirements for Aadhaar e-KYC?",
            "What is XML-based e-KYC and how is it processed?",
        ]
    },
    {
        "domain": "VKYC (Video KYC)",
        "subtopics": [
            "What is Video KYC (VKYC) as per RBI guidelines?",
            "What are the step-by-step requirements for VKYC?",
            "What documents are needed for VKYC?",
            "What is the role of the Video KYC officer?",
            "How is VKYC different from regular in-person KYC?",
            "What are the RBI guidelines on VKYC (master circular)?",
            "How to implement VKYC for a digital banking platform?",
        ]
    },
    {
        "domain": "DKYC (Digital KYC)",
        "subtopics": [
            "What is Digital KYC (DKYC) as per RBI guidelines?",
            "How does DKYC work and what are the technical requirements?",
            "What is the difference between DKYC and e-KYC?",
            "What are the RBI master circular guidelines for DKYC?",
            "How to register as a DKYC user?",
        ]
    },
    {
        "domain": "CERSAI",
        "subtopics": [
            "What is CERSAI (Central Registry of Securitisation Asset Reconstruction and Security Interest)?",
            "What is CERSAI registration and when is it required?",
            "How to file CERSAI charges online?",
            "What is the process for CERSAI satisfaction of charge?",
            "What are the fees and timelines for CERSAI registration?",
            "How does CERSAI integrate with MCA (Ministry of Corporate Affairs)?",
            "What is the difference between CERSAI and CIBIL?",
        ]
    },
    {
        "domain": "DMS (Document Management System)",
        "subtopics": [
            "What is a Document Management System (DMS) in banking?",
            "How does DMS work in loan processing?",
            "What are the key features of a financial DMS?",
            "How is DMS integrated with core banking systems?",
            "What are the compliance requirements for document storage in finance?",
            "What is the difference between DMS and record management?",
        ]
    },
    {
        "domain": "Regulatory Bodies & Compliance",
        "subtopics": [
            "What is the role of RBI in regulating Indian financial institutions?",
            "What is the role of SEBI in regulating capital markets?",
            "What is IRDAI and what does it regulate?",
            "What is the role of FIU-IND in financial intelligence?",
            "How do regulatory bodies coordinate for financial compliance?",
            "What are the penalties for non-compliance with financial regulations?",
        ]
    },
    {
        "domain": "General Fintech Concepts",
        "subtopics": [
            "What is a fintech company and how is it regulated in India?",
            "What is the difference between NBFC and a bank?",
            "What is a Payment Aggregator and Payment Gateway license?",
            "What is the PPI (Prepaid Payment Instruments) license?",
            "What is Account Aggregator framework in India?",
            "How does Open Banking work in India?",
            "What is UPI and how does it work technically?",
            "What is the RBI's Regulatory Sandbox?",
        ]
    },
]

SYSTEM_PROMPT = "You are an Indian fintech regulatory expert. Provide clear, accurate, and detailed explanations about Indian financial regulations, processes, and compliance requirements. Use simple language suitable for someone new to fintech."

INFERENCE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9,
    "api_max_tokens": 512,
}
