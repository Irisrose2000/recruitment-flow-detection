"""
Job Fraud Shield - Heuristic Safety Rules & Threat Indicators
============================================================
Deterministic regular-expression patterns targeting:
- Advance-fee & cashier check fraud
- Unrealistic pay for no-experience work
- Off-platform contact redirection (Telegram, WhatsApp, personal webmail)
- Urgent placement pressure
- Premature sensitive identity/banking data solicitation
"""

import re
from typing import List, Dict, Any

HEURISTIC_PATTERNS = [
    {
        'category': 'Financial Red Flag',
        'pattern': r'\b(wire transfer|western union|moneygram|cashier\s*check|send money|pay fee|advance fee|deposit\s*check|cryptocurrency|bitcoin)\b',
        'severity': 'HIGH',
        'detail': 'Mentions wire transfers, check cashing, advance fees, or cryptocurrency.',
        'tip': 'Legitimate employers never ask employees to deposit checks or wire funds for supplies.'
    },
    {
        'category': 'Unrealistic Compensation / No Experience',
        'pattern': r'\b(no experience (required|needed)|earn \$[4-9]\d{1,2}|earn \$[1-9]\d{3}|no interview|make money fast|instant cash)\b',
        'severity': 'MEDIUM',
        'detail': 'Promises very high pay with no prior experience or screening.',
        'tip': 'Unusually high compensation for entry-level work is a classic bait tactic.'
    },
    {
        'category': 'Unprofessional Contact Channel',
        'pattern': r'(@gmail\.com|@yahoo\.com|@hotmail\.com|telegram|whatsapp|google hangouts|text me on)\b',
        'severity': 'HIGH',
        'detail': 'Directs communication to free webmail (Gmail/Yahoo) or messaging apps (Telegram/WhatsApp).',
        'tip': 'Reputable hiring managers communicate via official corporate email domains.'
    },
    {
        'category': 'Urgent Placement Pressure',
        'pattern': r'\b(urgent hiring|immediate placement|start today|urgent vacancy|apply immediately|limited spots)\b',
        'severity': 'LOW',
        'detail': 'Appeals to urgency to rush candidates through normal due diligence.',
        'tip': 'Scammers use false urgency so victims make hasty decisions without verifying legitimacy.'
    },
    {
        'category': 'Sensitive Personal Info Request',
        'pattern': r'\b(social security|ssn|bank account details|routing number|credit card|date of birth)\b',
        'severity': 'HIGH',
        'detail': 'Solicits sensitive banking or identity credentials upfront.',
        'tip': 'Never provide banking credentials or SSN before receiving a formal, verified written offer.'
    }
]

def scan_risk_indicators(text: str) -> List[Dict[str, Any]]:
    """Scans text for deterministic employment scam patterns."""
    if not isinstance(text, str):
        return []
        
    text_lower = text.lower()
    indicators = []
    
    for item in HEURISTIC_PATTERNS:
        matches = [m.group(0) for m in re.finditer(item['pattern'], text_lower)]
        if matches:
            indicators.append({
                'category': item['category'],
                'severity': item['severity'],
                'detail': item['detail'],
                'tip': item['tip'],
                'matched': list(set(matches[:3]))
            })
            
    return indicators
