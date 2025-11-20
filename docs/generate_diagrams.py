"""
Generate System Architecture Diagrams
Creates visual diagrams for the NIDS system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_architecture_diagram():
    """Create system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'NIDS System Architecture', 
            ha='center', va='center', fontsize=20, weight='bold')
    
    # Network Logs
    box1 = FancyBboxPatch((3.5, 10), 3, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 10.4, 'Network Traffic Logs', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Preprocessing
    box2 = FancyBboxPatch((3.5, 8.5), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 8.9, 'Data Preprocessing', 
            ha='center', va='center', fontsize=12, weight='bold')
    ax.text(5, 8.6, 'Feature Engineering', 
            ha='center', va='center', fontsize=10)
    
    # Detection modules
    box3 = FancyBboxPatch((1, 6.5), 2.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(2.25, 7.5, 'Signature', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(2.25, 7.2, 'Detection', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(2.25, 6.9, 'Rule-based', ha='center', va='center', fontsize=9)
    ax.text(2.25, 6.7, 'patterns', ha='center', va='center', fontsize=9)
    
    box4 = FancyBboxPatch((3.75, 6.5), 2.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 7.5, 'Supervised ML', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(5, 7.2, 'RF, LR, XGBoost', ha='center', va='center', fontsize=9)
    ax.text(5, 6.9, 'Known threats', ha='center', va='center', fontsize=9)
    
    box5 = FancyBboxPatch((6.5, 6.5), 2.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(7.75, 7.5, 'Unsupervised ML', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(7.75, 7.2, 'Isolation Forest', ha='center', va='center', fontsize=9)
    ax.text(7.75, 6.9, 'Novel threats', ha='center', va='center', fontsize=9)
    
    # Hybrid Fusion
    box6 = FancyBboxPatch((3.5, 4.5), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(5, 4.9, 'Hybrid Fusion Algorithm', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Final Decision
    box7 = FancyBboxPatch((3.5, 3), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='lightsteelblue', edgecolor='black', linewidth=2)
    ax.add_patch(box7)
    ax.text(5, 3.4, 'Final Decision & Alerting', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Dashboard
    box8 = FancyBboxPatch((3.5, 1), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='lavender', edgecolor='black', linewidth=2)
    ax.add_patch(box8)
    ax.text(5, 1.4, 'Dashboard & Monitoring', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((5, 10), (5, 9.3), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((5, 8.5), (2.25, 7.3), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((5, 8.5), (5, 7.3), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((5, 8.5), (7.75, 7.3), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((2.25, 6.5), (4.2, 4.9), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((5, 6.5), (5, 5.3), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow6)
    
    arrow7 = FancyArrowPatch((7.75, 6.5), (5.8, 4.9), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    arrow8 = FancyArrowPatch((5, 4.5), (5, 3.8), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow8)
    
    arrow9 = FancyArrowPatch((5, 3), (5, 1.8), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow9)
    
    plt.tight_layout()
    plt.savefig('docs/architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to docs/architecture_diagram.png")


def create_decision_flow_diagram():
    """Create decision flow diagram for fusion algorithm"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Hybrid Fusion Decision Flow', 
            ha='center', va='center', fontsize=18, weight='bold')
    
    # Inputs
    box1 = FancyBboxPatch((1, 7.5), 2, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(2, 7.8, 'Signature', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2, 7.6, 'Confidence', ha='center', va='center', fontsize=9)
    
    box2 = FancyBboxPatch((7, 7.5), 2, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(8, 7.8, 'ML', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(8, 7.6, 'Confidence', ha='center', va='center', fontsize=9)
    
    # Decision diamond
    diamond = mpatches.FancyBboxPatch((3.5, 6), 3, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='yellow', edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 7, 'sig_conf > 0.8?', ha='center', va='center', fontsize=10, weight='bold')
    
    # High confidence path
    box3 = FancyBboxPatch((0.5, 4), 2.5, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(1.75, 4.4, 'Trust Signature', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(1.75, 4.1, 'Attack', ha='center', va='center', fontsize=9)
    
    # Medium confidence path
    diamond2 = mpatches.FancyBboxPatch((3.5, 4), 3, 1.5,
                                       boxstyle="round,pad=0.1",
                                       facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(diamond2)
    ax.text(5, 5, 'sig_conf > 0.5?', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(5, 4.7, 'Check ML', ha='center', va='center', fontsize=9)
    
    # ML check
    diamond3 = mpatches.FancyBboxPatch((7, 4), 2.5, 1.5,
                                       boxstyle="round,pad=0.1",
                                       facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(diamond3)
    ax.text(8.25, 5, 'ml_conf > 0.6?', ha='center', va='center', fontsize=10, weight='bold')
    
    # Low confidence path
    box4 = FancyBboxPatch((3.5, 2), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 2.4, 'Trust ML', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(5, 2.1, 'ml_conf > 0.7?', ha='center', va='center', fontsize=9)
    
    # Outputs
    box5 = FancyBboxPatch((0.5, 0.5), 2, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='red', edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(1.5, 0.8, 'ATTACK', ha='center', va='center', fontsize=11, weight='bold')
    
    box6 = FancyBboxPatch((4, 0.5), 2, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='green', edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(5, 0.8, 'NORMAL', ha='center', va='center', fontsize=11, weight='bold')
    
    # Arrows
    arrows = [
        ((2, 7.5), (4.2, 7.2)),
        ((8, 7.5), (5.8, 7.2)),
        ((5, 6), (1.75, 4.8)),
        ((5, 6), (5, 5.5)),
        ((5, 4), (8.25, 5.2)),
        ((8.25, 4), (1.5, 1.1)),
        ((5, 4), (5, 2.8)),
        ((5, 2), (1.5, 1.1)),
        ((5, 2), (5, 1.1)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('docs/decision_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("Decision flow diagram saved to docs/decision_flow_diagram.png")


if __name__ == '__main__':
    print("Generating diagrams...")
    create_architecture_diagram()
    create_decision_flow_diagram()
    print("Done!")

