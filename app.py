from flask import Flask, request, jsonify, render_template, send_file
import PyPDF2
import spacy
import re
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load advanced NLP models
nlp = spacy.load("en_core_web_lg")

# Comprehensive job database with skills and descriptions
jobs_db = {
    "Data Scientist": {
        "skills": ["machine learning", "python", "data analysis", "statistics", "tensorflow", "keras", "pandas", "numpy"],
        "jobs": ["Data Scientist at XYZ Tech", "AI Researcher at InnovAI", "Machine Learning Engineer at DataCorp"],
        "typical_responsibilities": [
            "Develop machine learning models",
            "Analyze complex datasets",
            "Create predictive analytics solutions",
            "Implement data-driven strategies"
        ]
    },
    "Software Engineer": {
        "skills": ["python", "java", "javascript", "react", "node.js", "docker", "kubernetes", "sql"],
        "jobs": ["Backend Developer at TechGiant", "Full Stack Engineer at WebSolutions", "DevOps Engineer at CloudNative"],
        "typical_responsibilities": [
            "Design and develop scalable software systems",
            "Implement new features and improvements",
            "Collaborate with cross-functional teams",
            "Ensure code quality and performance"
        ]
    },
    "Project Manager": {
        "skills": ["agile", "scrum", "project planning", "communication", "jira", "microsoft project", "risk management"],
        "jobs": ["IT Project Manager at GlobalCorp", "Program Coordinator at TechInnovate", "Scrum Master at Agile Solutions"],
        "typical_responsibilities": [
            "Coordinate project timelines and resources",
            "Manage stakeholder expectations",
            "Implement agile methodologies",
            "Monitor project progress and risks"
        ]
    },
    "Marketing Specialist": {
        "skills": ["digital marketing", "social media", "content creation", "analytics", "seo", "google analytics"],
        "jobs": ["Digital Marketing Coordinator at BrandHub", "Social Media Strategist at MediaWave"],
        "typical_responsibilities": [
            "Develop marketing campaigns",
            "Analyze market trends",
            "Create engaging content",
            "Manage social media channels"
        ]
    }
}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with improved parsing"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def advanced_skills_extraction(text):
    """Extract skills and professional details using NLP"""
    doc = nlp(text.lower())
    
    # Extract named entities
    entities = {ent.text: ent.label_ for ent in doc.ents}
    
    # Extract potential skills and keywords
    skills = set()
    job_keywords = set()
    
    # Check against predefined job databases
    for job, job_data in jobs_db.items():
        matched_skills = [skill for skill in job_data['skills'] if skill in text.lower()]
        skills.update(matched_skills)
        
        if len(matched_skills) > 2:
            job_keywords.add(job)
    
    # Additional skill extraction using custom patterns
    skill_patterns = [
        r'\b(proficient|expertise|experienced)\s+in\s+([a-zA-Z\s]+)',
        r'\b(skilled|knowledge)\s+of\s+([a-zA-Z\s]+)',
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.update([match[1].strip() for match in matches])
    
    return {
        'detected_skills': list(skills),
        'potential_job_titles': list(job_keywords),
        'named_entities': entities
    }

def match_jobs_with_resume(skills, job_database=jobs_db):
    """Match resume skills with potential job opportunities"""
    matched_jobs = []
    skills = [skill.lower() for skill in skills]
    
    for job, job_data in job_database.items():
        job_skills = [skill.lower() for skill in job_data.get('skills', [])]
        
        # Calculate skill match percentage
        matched_skill_count = len(set(skills) & set(job_skills))
        skill_match_percentage = (matched_skill_count / len(job_skills)) * 100 if job_skills else 0
        
        if skill_match_percentage > 50:
            matched_jobs.append({
                'job_title': job,
                'match_percentage': round(skill_match_percentage, 2),
                'available_positions': job_data.get('jobs', []),
                'typical_responsibilities': job_data.get('typical_responsibilities', [])
            })
    
    return sorted(matched_jobs, key=lambda x: x['match_percentage'], reverse=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    # Extract text from PDF
    text = extract_text_from_pdf(file)
    
    # Advanced skills and job title extraction
    analysis = advanced_skills_extraction(text)
    
    # Match jobs based on extracted skills
    matched_jobs = match_jobs_with_resume(analysis['detected_skills'])
    
    return jsonify({
        'resume_content': text[:2000],  # First 2000 characters for preview
        'job_titles': analysis['potential_job_titles'],
        'detected_skills': analysis['detected_skills'],
        'named_entities': list(analysis['named_entities'].keys()),
        'jobs': [job['job_title'] for job in matched_jobs],
        'detailed_job_matches': matched_jobs
    })

@app.route('/full-resume', methods=['POST'])
def get_full_resume():
    """Route to serve full resume if needed"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    return send_file(
        io.BytesIO(file.read()),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='uploaded_resume.pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)