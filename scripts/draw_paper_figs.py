import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# Set academic style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Color palette (Nature-like)
COLORS = {
    'primary': '#E64B35',  # Red
    'secondary': '#4DBBD5', # Blue
    'tertiary': '#00A087',  # Green
    'quaternary': '#3C5488', # Dark Blue
    'gray': '#7E6148',      # Brown/Gray
    'light_bg': '#F0F0F0',
    'box_fill': '#FFFFFF'
}

def draw_rounded_box(ax, x, y, w, h, text, color='black', fill_color='white', lw=1.5, fontsize=10):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                 linewidth=lw, edgecolor=color, facecolor=fill_color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return box

def draw_arrow_custom(ax, x1, y1, x2, y2, color='black', style='->'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5, shrinkA=0, shrinkB=0))

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # User
    draw_rounded_box(ax, 0.5, 4.5, 1.5, 1, "User", color=COLORS['quaternary'], fill_color='#E8F0FE')
    
    # LLM
    draw_rounded_box(ax, 8, 4.5, 1.5, 1, "LLM", color=COLORS['quaternary'], fill_color='#E8F0FE')

    # LightMem System Boundary
    rect = patches.Rectangle((2.5, 0.5), 5, 5, linewidth=1, edgecolor='gray', facecolor='#F9F9F9', linestyle='--')
    ax.add_patch(rect)
    ax.text(5, 5.2, "LightMem System", ha='center', fontsize=12, fontweight='bold', color='gray')

    # Controller / Memory Manager
    draw_rounded_box(ax, 3.5, 2.5, 3, 1, "Memory Manager\n(Controller)", color=COLORS['primary'], fill_color='#FFF5F5')

    # Components
    # Sensory Memory
    draw_rounded_box(ax, 3.5, 4.0, 3, 0.8, "Sensory Memory\n(Short-term Buffer)", color=COLORS['secondary'], fill_color='#E0F7FA')
    
    # Retriever
    draw_rounded_box(ax, 3.0, 1.0, 1.8, 0.8, "Retriever", color=COLORS['tertiary'], fill_color='#E8F5E9')
    
    # Storage
    draw_rounded_box(ax, 5.2, 1.0, 1.8, 0.8, "Vector DB\n(Long-term)", color=COLORS['gray'], fill_color='#EFEBE9')

    # Arrows
    # User -> Sensory
    draw_arrow_custom(ax, 2.0, 5.0, 3.5, 4.4, color='black') # User to Sensory
    ax.text(2.7, 4.8, "Query", fontsize=9, ha='center')

    # Sensory -> Manager
    draw_arrow_custom(ax, 5.0, 4.0, 5.0, 3.5, color='black')

    # Manager -> Retriever
    draw_arrow_custom(ax, 4.0, 2.5, 4.0, 1.8, color='black')
    
    # Retriever <-> Storage
    draw_arrow_custom(ax, 4.8, 1.4, 5.2, 1.4, style='<->', color='gray')

    # Manager -> LLM (Constructed Prompt)
    draw_arrow_custom(ax, 6.5, 3.0, 8.0, 4.5, color='black')
    ax.text(7.2, 3.5, "Augmented\nPrompt", fontsize=9, ha='center', rotation=25)

    # LLM -> User
    draw_arrow_custom(ax, 8.0, 5.0, 2.0, 5.0, style='->', color='black') # Back to User? Or LLM -> Manager -> User?
    # Actually LLM -> Manager (History)
    draw_arrow_custom(ax, 8.0, 4.8, 6.5, 4.4, style='->', color='gray') # Response loop
    ax.text(7.2, 4.8, "Response", fontsize=9, ha='center')

    # Manager -> Storage (Update)
    draw_arrow_custom(ax, 6.0, 2.5, 6.0, 1.8, color='black')
    ax.text(6.3, 2.1, "Update", fontsize=8)

    plt.tight_layout()
    plt.savefig('figs/fig1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figs/fig1_architecture.png")

def fig_mechanism():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    y_center = 2

    # Step 1: Input
    draw_rounded_box(ax, 0.5, 1.5, 2, 1, "Conversation\nStream", color=COLORS['quaternary'], fill_color='#E8F0FE')
    
    # Arrow
    draw_arrow_custom(ax, 2.5, 2, 3, 2)

    # Step 2: Segmentation
    draw_rounded_box(ax, 3, 1.5, 2, 1, "Topic\nSegmentation", color=COLORS['secondary'], fill_color='#E0F7FA')
    ax.text(4, 1.2, "(LLMLingua2)", fontsize=8, ha='center', color='gray')

    # Arrow
    draw_arrow_custom(ax, 5, 2, 5.5, 2)

    # Step 3: Compression
    draw_rounded_box(ax, 5.5, 1.5, 2, 1, "Information\nCompression", color=COLORS['primary'], fill_color='#FFF5F5')
    ax.text(6.5, 1.2, "(Entropy / KV)", fontsize=8, ha='center', color='gray')

    # Arrow
    draw_arrow_custom(ax, 7.5, 2, 8, 2)

    # Step 4: Embedding
    draw_rounded_box(ax, 8, 1.5, 1.5, 1, "Embedding", color=COLORS['tertiary'], fill_color='#E8F5E9')

    # Arrow
    draw_arrow_custom(ax, 9.5, 2, 10, 2)

    # Step 5: Storage
    draw_rounded_box(ax, 10, 1.5, 1.5, 1, "Qdrant\nStorage", color=COLORS['gray'], fill_color='#EFEBE9')

    plt.tight_layout()
    plt.savefig('figs/fig2_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figs/fig2_mechanism.png")

def fig_performance():
    # Mock Data
    methods = ['RAG Baseline', 'Mem0', 'LangMem', 'LightMem (Ours)']
    accuracy = [0.65, 0.72, 0.75, 0.82]
    latency = [120, 150, 180, 130] # ms

    x = np.arange(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar 1: Accuracy
    bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=COLORS['secondary'], alpha=0.9)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=COLORS['secondary'])
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor=COLORS['secondary'])
    
    # Bar 2: Latency (Secondary Axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, latency, width, label='Latency (ms)', color=COLORS['primary'], alpha=0.9)
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax2.set_ylim(0, 250)
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])

    # Labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax1.set_title('Performance Comparison on Long-Context Tasks', fontsize=14, pad=20)

    # Grid
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('figs/fig3_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figs/fig3_performance.png")

def fig_entropy_principle():
    """
    Visualizes the entropy-based token selection mechanism.
    """
    # Sample sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    words = sentence.split()
    # Mock entropy values (High for content words, low for stop words)
    # The, quick, brown, fox, jumps, over, the, lazy, dog
    entropy = [0.5, 4.2, 3.8, 5.1, 4.5, 1.2, 0.4, 3.9, 4.8]
    threshold = 2.5
    
    selected_indices = [i for i, e in enumerate(entropy) if e >= threshold]
    selected_words = [words[i] for i in selected_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Bar chart of entropy
    x = np.arange(len(words))
    colors = [COLORS['primary'] if e >= threshold else COLORS['gray'] for e in entropy]
    
    bars = ax1.bar(x, entropy, color=colors, alpha=0.8, width=0.6)
    
    # Threshold line
    ax1.axhline(y=threshold, color=COLORS['secondary'], linestyle='--', linewidth=2, label='Selection Threshold')
    ax1.text(len(words)-0.5, threshold+0.1, 'Threshold', color=COLORS['secondary'], va='bottom', ha='right', fontsize=10, fontweight='bold')
    
    # Labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(words, fontsize=11)
    ax1.set_ylabel('Self-Information (Entropy)', fontsize=12)
    ax1.set_title('Token Importance Estimation', fontsize=14, pad=15)
    ax1.set_ylim(0, 6)
    
    # Add values on top of bars
    for bar, val in zip(bars, entropy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

    # Plot 2: Compression Result
    ax2.axis('off')
    
    # Draw original text
    ax2.text(0.1, 0.7, "Original:", fontsize=12, fontweight='bold', color='black')
    for i, word in enumerate(words):
        color = COLORS['primary'] if entropy[i] >= threshold else 'gray'
        weight = 'bold' if entropy[i] >= threshold else 'normal'
        ax2.text(0.25 + i*0.08, 0.7, word, fontsize=12, color=color, fontweight=weight)
        
    # Draw arrow
    draw_arrow_custom(ax2, 0.5, 0.55, 0.5, 0.35, color='black')
    ax2.text(0.52, 0.45, "Compress (Top-k%)", fontsize=10)

    # Draw compressed text
    ax2.text(0.1, 0.2, "Compressed:", fontsize=12, fontweight='bold', color='black')
    
    # Reconstruct the string with spacing
    compressed_str = " ".join(selected_words)
    draw_rounded_box(ax2, 0.25, 0.1, 0.6, 0.2, compressed_str, color=COLORS['primary'], fill_color='#FFF5F5', fontsize=12)

    plt.tight_layout()
    plt.savefig('figs/fig4_entropy_principle.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figs/fig4_entropy_principle.png")

def fig_memory_lifecycle():
    """
    Visualizes the lifecycle of a memory entry: Sensory -> Processing -> Long-term -> Retrieval
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Central Hub: LightMem Controller
    draw_rounded_box(ax, 4, 3, 2, 1, "LightMem\nController", color=COLORS['quaternary'], fill_color='white', lw=2)
    
    # 1. Sensory Memory (Top Left)
    draw_rounded_box(ax, 1, 5, 2.5, 1.2, "Sensory Buffer\n(Raw Dialogue)", color=COLORS['secondary'], fill_color='#E0F7FA')
    ax.text(2.25, 6.4, "1. Accumulate", fontsize=11, fontweight='bold', ha='center', color=COLORS['secondary'])
    
    # 2. Processor (Top Right)
    draw_rounded_box(ax, 6.5, 5, 2.5, 1.2, "Processor\n(Segment & Compress)", color=COLORS['primary'], fill_color='#FFF5F5')
    ax.text(7.75, 6.4, "2. Consolidate", fontsize=11, fontweight='bold', ha='center', color=COLORS['primary'])
    
    # 3. Long-term Storage (Bottom Right)
    draw_rounded_box(ax, 6.5, 0.8, 2.5, 1.2, "Vector DB\n(Qdrant)", color=COLORS['gray'], fill_color='#EFEBE9')
    ax.text(7.75, 0.5, "3. Store", fontsize=11, fontweight='bold', ha='center', color=COLORS['gray'])
    
    # 4. Retrieval (Bottom Left)
    draw_rounded_box(ax, 1, 0.8, 2.5, 1.2, "Retriever\n(Semantic Search)", color=COLORS['tertiary'], fill_color='#E8F5E9')
    ax.text(2.25, 0.5, "4. Retrieve", fontsize=11, fontweight='bold', ha='center', color=COLORS['tertiary'])
    
    # Arrows (Cycle)
    # Sensory -> Controller
    draw_arrow_custom(ax, 2.25, 5, 4, 3.8, color='gray')
    
    # Controller -> Processor
    draw_arrow_custom(ax, 5, 4, 6.5, 5.2, color='gray')
    
    # Processor -> Storage
    draw_arrow_custom(ax, 7.75, 5, 7.75, 2, color='gray')
    ax.text(8.0, 3.5, "Embed", fontsize=9, rotation=270)
    
    # Storage -> Controller (via Retrieval)
    # Actually Retrieval queries Storage
    draw_arrow_custom(ax, 3.5, 1.4, 6.5, 1.4, style='<->', color='gray')
    ax.text(5, 1.6, "Query", fontsize=9, ha='center')
    
    # Retriever -> Controller
    draw_arrow_custom(ax, 2.25, 2, 4, 3.2, color='gray')
    
    # Controller -> Output
    draw_arrow_custom(ax, 5, 3, 5, 1.8, style='->', color='black') # Logic link
    
    # User Input to Sensory
    draw_arrow_custom(ax, 0.5, 5.6, 1, 5.6, color='black', style='->')
    ax.text(0.2, 5.6, "Input", fontsize=10, ha='left')
    
    # Retrieval to LLM
    draw_arrow_custom(ax, 4, 3, 4, 4.5, color='black', style='->')
    ax.text(3.5, 4.2, "To LLM", fontsize=10)

    plt.title("Memory Lifecycle in LightMem", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig('figs/fig5_lifecycle.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figs/fig5_lifecycle.png")

def main():
    print("Generating figures...")
    fig_architecture()
    fig_mechanism()
    fig_performance()
    fig_entropy_principle()
    fig_memory_lifecycle()
    print("Done. Figures saved in 'figs/' directory.")

if __name__ == "__main__":
    main()
