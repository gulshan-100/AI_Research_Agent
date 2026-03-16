"""
Evaluation script for AI Research Agent system
Tests both retrieval quality and generation quality using quantitative metrics

This script provides:
1. Retrieval metrics: Precision, Recall, MRR, NDCG
2. Generation metrics: ROUGE, BERTScore, Factual consistency
3. Performance metrics: Latency, throughput
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
import django
django.setup()

# Import agent components
from agent.document_loader import DocumentProcessor
from agent.research_agents.it_agent import ITResearchAgent
from agent.research_agents.pharma_agent import PharmaResearchAgent
from agent.research_agents.selector import AgentSelector

# For evaluation
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ground truth data - manually labeled relevant documents for test queries
# Format: {query: [list of document IDs that are relevant]}
GROUND_TRUTH_IT = {
    "What are the major trends in the Indian IT industry?": [
        "IT-industry-outlook-aug23.pdf",
        "INTRODUCTION-TO-INDIAN-SOFTWARE-INDUSTRY-IN-GLOBAL-PERSPECTIV.pdf",
        "Indian ICT Sectorial System of Innovation (IISSI) Report_0.pdf"
    ],
    "What is the size of Indian IT market?": [
        "it-and-ites-sector-risk-report-2022.pdf",
        "it-_ites_report_28_jan.pdf",
        "IT-industry-outlook-aug23.pdf"
    ],
    "What are the challenges faced by Indian IT companies?": [
        "technology-sector-trends-and-priorities.pdf",
        "Causes_and_Consequences_of_IT_Boom.pdf",
        "mathur.pdf"
    ]
}

GROUND_TRUTH_PHARMA = {
    "What are the major trends in the Indian pharmaceutical industry?": [
        "Drugsand Pharmaceuticals Industry-India-Nov2024.pdf",
        "Indianpharmasector_currentstatus.pdf"
    ],
    "What is the size of Indian pharmaceutical market?": [
        "Drugsand Pharmaceuticals Industry-India-Nov2024.pdf",
        "doc2024822379301.pdf"
    ],
    "What are the challenges faced by Indian pharmaceutical companies?": [
        "doc2025518556901.pdf",
        "Indianpharmasector_currentstatus.pdf"
    ]
}

# Define reference summaries for ROUGE evaluation
REFERENCE_SUMMARIES = {
    "Indian IT industry": """
    The Indian IT industry has shown remarkable growth over the past decade, becoming a global leader
    in software services and IT-enabled services (ITeS). Key trends include cloud computing adoption,
    digital transformation services, AI and ML integration, and cybersecurity solutions. The industry
    faces challenges including talent shortages, visa restrictions, and increasing competition from
    other countries. Future growth areas include edge computing, blockchain applications, and
    specialized AI solutions.
    """,
    
    "Indian pharmaceutical industry": """
    The Indian pharmaceutical industry is one of the largest global suppliers of generic medicines,
    accounting for 20% of global generics exports by volume. Key trends include increasing R&D spending,
    biosimilar development, and API manufacturing. The industry faces challenges including price controls,
    regulatory scrutiny, and quality concerns. Future growth areas include complex generics, specialty
    pharmaceuticals, and contract research and manufacturing services (CRAMS).
    """
}

class EvaluationMetrics:
    """
    Calculate and report evaluation metrics for retrieval and generation tasks
    """
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate precision@k: What fraction of retrieved documents are relevant
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top documents to consider
            
        Returns:
            Precision@k score (0-1)
        """
        if not retrieved_docs or k <= 0:
            return 0.0
            
        # Consider only top-k documents
        retrieved_k = retrieved_docs[:min(k, len(retrieved_docs))]
        
        # Count relevant documents in the retrieved set
        relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
        
        # Calculate precision
        return len(relevant_retrieved) / len(retrieved_k)
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate recall@k: What fraction of relevant documents are retrieved
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top documents to consider
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant_docs or k <= 0:
            return 0.0
            
        # Consider only top-k documents
        retrieved_k = retrieved_docs[:min(k, len(retrieved_docs))]
        
        # Count relevant documents in the retrieved set
        relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
        
        # Calculate recall
        return len(relevant_retrieved) / len(relevant_docs)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR): The inverse of the rank of the first relevant document
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            
        Returns:
            MRR score (0-1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
            
        # Find the first relevant document
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                # +1 because rank is 1-based
                return 1.0 / (i + 1)
                
        # No relevant documents found
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top documents to consider
            
        Returns:
            NDCG@k score (0-1)
        """
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0
            
        # Binary relevance: 1 if document is relevant, 0 otherwise
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            # i+2 because i is 0-indexed and log base 2
            dcg += rel / np.log2(i + 2)
            
        # Calculate ideal DCG (perfect ranking)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
            
        # NDCG is DCG / IDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_rouge(generated_text: str, reference_text: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE metrics for generated text
        
        Args:
            generated_text: Generated text to evaluate
            reference_text: Reference text to compare against
            
        Returns:
            Dictionary of ROUGE scores
        """
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate ROUGE scores
        scores = scorer.score(reference_text, generated_text)
        
        # Convert scores to dictionary
        result = {}
        for metric, score in scores.items():
            result[metric] = {
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            }
            
        return result
    
    @staticmethod
    def calculate_latency(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Calculate latency of a function call
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (function result, latency in seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return result, end_time - start_time


class RetrieverEvaluator:
    """
    Evaluate the retrieval component of the system
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.metrics = EvaluationMetrics()
        
    def extract_doc_id(self, doc) -> str:
        """Extract document ID from a retrieved document"""
        # Extract filename from metadata or use content hash
        return doc.metadata.get('filename', 'unknown')
        
    def evaluate_query(self, query: str, sector: str, ground_truth: List[str], 
                       k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query
        
        Args:
            query: Query string
            sector: Sector (IT or Pharma)
            ground_truth: List of relevant document IDs
            k: Number of documents to retrieve
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get documents from retriever
        retrieved_docs, latency = self.metrics.calculate_latency(
            self.processor.get_relevant_documents,
            query=query, sector=sector, k=k
        )
        
        # Extract document IDs
        retrieved_ids = [self.extract_doc_id(doc) for doc in retrieved_docs]
        
        # Calculate metrics
        precision = self.metrics.precision_at_k(retrieved_ids, ground_truth, k)
        recall = self.metrics.recall_at_k(retrieved_ids, ground_truth, k)
        mrr = self.metrics.mean_reciprocal_rank(retrieved_ids, ground_truth)
        ndcg = self.metrics.ndcg_at_k(retrieved_ids, ground_truth, k)
        
        return {
            'precision@k': precision,
            'recall@k': recall,
            'mrr': mrr,
            'ndcg@k': ndcg,
            'latency': latency,
            'retrieved_docs': retrieved_ids,
            'relevant_docs': ground_truth
        }
    
    def evaluate_sector(self, queries: Dict[str, List[str]], sector: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval for all queries in a sector
        
        Args:
            queries: Dictionary of {query: relevant_docs}
            sector: Sector name
            
        Returns:
            Dictionary of {query: metrics}
        """
        results = {}
        
        for query, relevant_docs in queries.items():
            logger.info(f"Evaluating query: {query}")
            metrics = self.evaluate_query(query, sector, relevant_docs)
            results[query] = metrics
            
        return results
    
    def evaluate_all(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate retrieval for all sectors
        
        Returns:
            Dictionary of {sector: {query: metrics}}
        """
        results = {
            'IT': self.evaluate_sector(GROUND_TRUTH_IT, 'IT'),
            'Pharma': self.evaluate_sector(GROUND_TRUTH_PHARMA, 'Pharma')
        }
        
        # Calculate average metrics per sector
        for sector, queries in results.items():
            avg_precision = np.mean([q['precision@k'] for q in queries.values()])
            avg_recall = np.mean([q['recall@k'] for q in queries.values()])
            avg_mrr = np.mean([q['mrr'] for q in queries.values()])
            avg_ndcg = np.mean([q['ndcg@k'] for q in queries.values()])
            avg_latency = np.mean([q['latency'] for q in queries.values()])
            
            results[sector]['__avg__'] = {
                'precision@k': avg_precision,
                'recall@k': avg_recall,
                'mrr': avg_mrr,
                'ndcg@k': avg_ndcg,
                'latency': avg_latency
            }
            
        return results


class GenerationEvaluator:
    """
    Evaluate the generation component of the system
    """
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
        self.it_agent = ITResearchAgent()
        self.pharma_agent = PharmaResearchAgent()
        self.agent_selector = AgentSelector()
        
    def evaluate_generation(self, topic: str, reference_summary: str) -> Dict[str, Any]:
        """
        Evaluate report generation for a topic
        
        Args:
            topic: Research topic
            reference_summary: Reference summary for ROUGE evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Select appropriate agent
        agent_type = self.agent_selector.select_agent(topic)
        agent = self.it_agent if agent_type == 'IT' else self.pharma_agent
        
        # Generate report
        report, latency = self.metrics.calculate_latency(
            agent.research,
            topic
        )
        
        # Calculate ROUGE metrics
        rouge_scores = self.metrics.calculate_rouge(report.content, reference_summary)
        
        return {
            'agent': agent_type,
            'rouge_scores': rouge_scores,
            'report_length': len(report.content),
            'num_sources': len(report.sources),
            'latency': latency
        }
    
    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate generation for all topics
        
        Returns:
            Dictionary of {topic: metrics}
        """
        results = {}
        
        for topic, reference in REFERENCE_SUMMARIES.items():
            logger.info(f"Evaluating generation for topic: {topic}")
            metrics = self.evaluate_generation(topic, reference)
            results[topic] = metrics
            
        return results


def plot_results(retrieval_results, generation_results, output_dir='evaluation_results'):
    """
    Generate plots of evaluation results
    
    Args:
        retrieval_results: Results from retriever evaluation
        generation_results: Results from generation evaluation
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot retrieval metrics by sector
    sectors = list(retrieval_results.keys())
    metrics = ['precision@k', 'recall@k', 'mrr', 'ndcg@k']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sectors))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [retrieval_results[sector]['__avg__'][metric] for sector in sectors]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Sector')
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Metrics by Sector')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(sectors)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_metrics_by_sector.png'))
    
    # Plot ROUGE scores for generation
    topics = list(generation_results.keys())
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(topics))
    width = 0.25
    
    for i, rouge_type in enumerate(rouge_types):
        values = [generation_results[topic]['rouge_scores'][rouge_type]['fmeasure'] for topic in topics]
        ax.bar(x + i*width, values, width, label=rouge_type)
    
    ax.set_xlabel('Topic')
    ax.set_ylabel('F-measure')
    ax.set_title('ROUGE Scores by Topic')
    ax.set_xticks(x + width * (len(rouge_types) - 1) / 2)
    ax.set_xticklabels(topics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rouge_scores_by_topic.png'))
    
    # Plot generation latency
    latencies = [generation_results[topic]['latency'] for topic in topics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(topics, latencies)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Generation Latency by Topic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_latency_by_topic.png'))


def main():
    """
    Run full system evaluation
    """
    logger.info("Starting AI Research Agent evaluation")
    
    try:
        # Evaluate retrieval
        logger.info("Evaluating retrieval component...")
        retriever_evaluator = RetrieverEvaluator()
        retrieval_results = retriever_evaluator.evaluate_all()
        
        # Log retrieval results
        for sector, queries in retrieval_results.items():
            if sector == '__avg__':
                continue
                
            logger.info(f"\n=== {sector} Sector Retrieval Results ===")
            
            for query, metrics in queries.items():
                if query == '__avg__':
                    logger.info(f"\nAverage metrics for {sector}:")
                    logger.info(f"Precision@5: {metrics['precision@k']:.3f}")
                    logger.info(f"Recall@5: {metrics['recall@k']:.3f}")
                    logger.info(f"MRR: {metrics['mrr']:.3f}")
                    logger.info(f"NDCG@5: {metrics['ndcg@k']:.3f}")
                    logger.info(f"Latency: {metrics['latency']:.3f}s")
                else:
                    logger.info(f"\nQuery: {query}")
                    logger.info(f"Precision@5: {metrics['precision@k']:.3f}")
                    logger.info(f"Recall@5: {metrics['recall@k']:.3f}")
                    logger.info(f"MRR: {metrics['mrr']:.3f}")
                    logger.info(f"NDCG@5: {metrics['ndcg@k']:.3f}")
                    logger.info(f"Retrieved: {metrics['retrieved_docs']}")
                    logger.info(f"Relevant: {metrics['relevant_docs']}")
        
        # Save retrieval results to file
        with open('retrieval_results.json', 'w') as f:
            json.dump(retrieval_results, f, indent=2)
        
        # Evaluate generation
        logger.info("\nEvaluating generation component...")
        generation_evaluator = GenerationEvaluator()
        generation_results = generation_evaluator.evaluate_all()
        
        # Log generation results
        for topic, metrics in generation_results.items():
            logger.info(f"\n=== {topic} Generation Results ===")
            logger.info(f"Agent: {metrics['agent']}")
            
            for rouge_type, scores in metrics['rouge_scores'].items():
                logger.info(f"{rouge_type} - P: {scores['precision']:.3f}, R: {scores['recall']:.3f}, F: {scores['fmeasure']:.3f}")
                
            logger.info(f"Report length: {metrics['report_length']} characters")
            logger.info(f"Number of sources: {metrics['num_sources']}")
            logger.info(f"Generation latency: {metrics['latency']:.3f}s")
        
        # Save generation results to file
        with open('generation_results.json', 'w') as f:
            json.dump(generation_results, f, indent=2)
        
        # Plot results
        logger.info("\nGenerating evaluation plots...")
        plot_results(retrieval_results, generation_results)
        
        logger.info("\nEvaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    main()
