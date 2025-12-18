import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=10):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return (x + width/2, y + height/2, x + width/2, y, x + width/2, y + height) # center, top_mid, bottom_mid

def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))

def generate_project_structure_img(filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Nodes
    # Main container
    draw_box(ax, 1, 1, 10, 8, "", color='whitesmoke')
    ax.text(6, 9.2, "src/lightmem", ha='center', fontsize=14, weight='bold')

    # Memory (Core)
    mem_c = draw_box(ax, 4.5, 6, 3, 1.5, "memory\n(Core Logic)", color='lightgreen')
    
    # Configs
    conf_c = draw_box(ax, 1.5, 6, 2, 1, "configs\n(Configuration)", color='lightyellow')
    
    # Factory
    fac_c = draw_box(ax, 4.5, 3.5, 3, 1.5, "factory\n(Component Factory)", color='lightsalmon')
    
    # Components (Abstract representation)
    comp_c = draw_box(ax, 4.5, 1.5, 3, 1, "Components Implementation\n(Manager, Retriever, etc.)", color='lightblue')
    
    # Toolkits
    tool_c = draw_box(ax, 8.5, 4, 2, 3, "memory_toolkits\n(Utils & Baselines)", color='thistle')

    # Edges
    # Memory -> Configs
    draw_arrow(ax, (mem_c[0]-1.5, mem_c[1]), (conf_c[0]+1, conf_c[1]))
    # Memory -> Factory
    draw_arrow(ax, (mem_c[2], mem_c[3]), (fac_c[4], fac_c[5]))
    # Factory -> Components
    draw_arrow(ax, (fac_c[2], fac_c[3]), (comp_c[4], comp_c[5]))
    
    # Toolkits -> Memory (loosely coupled, maybe extension)
    # Just show they exist in parallel
    
    plt.title("LightMem Project Structure", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def generate_class_diagram_img(filename):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # LightMemory Class
    lm_c = draw_box(ax, 5, 8, 4, 3, "LightMemory\n\n+config\n+manager\n+compressor\n+segmenter\n+retrievers\n\n+add_memory()\n+retrieve()", color='lightgreen')

    # Factories
    mf_c = draw_box(ax, 1, 5, 2, 1, "MemoryManager\nFactory", color='lightsalmon')
    pf_c = draw_box(ax, 3.5, 5, 2, 1, "PreCompressor\nFactory", color='lightsalmon')
    tf_c = draw_box(ax, 6, 5, 2, 1, "TopicSegmenter\nFactory", color='lightsalmon')
    rf_c = draw_box(ax, 8.5, 5, 2, 1, "Retriever\nFactory", color='lightsalmon')
    ef_c = draw_box(ax, 11, 5, 2, 1, "TextEmbedder\nFactory", color='lightsalmon')

    # Components (Implementations)
    mm_c = draw_box(ax, 1, 2, 2, 1, "MemoryManager\n(OpenAI, DeepSeek...)", color='lightblue')
    pc_c = draw_box(ax, 3.5, 2, 2, 1, "PreCompressor\n(Entropy, LLMLingua2)", color='lightblue')
    ts_c = draw_box(ax, 6, 2, 2, 1, "TopicSegmenter\n(LLMLingua2)", color='lightblue')
    rt_c = draw_box(ax, 8.5, 2, 2, 1, "Retrievers\n(BM25, Qdrant)", color='lightblue')
    te_c = draw_box(ax, 11, 2, 2, 1, "TextEmbedder\n(HuggingFace)", color='lightblue')

    # Edges from LightMemory to Factories (Use dependency)
    # Just drawing simple lines to show connection
    # LM -> MM Factory
    draw_arrow(ax, (lm_c[0]-1, lm_c[3]), (mf_c[4]+0.5, mf_c[5]))
    # LM -> PC Factory
    draw_arrow(ax, (lm_c[0]-0.5, lm_c[3]), (pf_c[4], pf_c[5]))
    # LM -> TS Factory
    draw_arrow(ax, (lm_c[2], lm_c[3]), (tf_c[4], tf_c[5]))
    # LM -> Ret Factory
    draw_arrow(ax, (lm_c[0]+0.5, lm_c[3]), (rf_c[4], rf_c[5]))
    # LM -> Emb Factory
    draw_arrow(ax, (lm_c[0]+1, lm_c[3]), (ef_c[4]-0.5, ef_c[5]))

    # Edges from Factories to Components (Create)
    draw_arrow(ax, (mf_c[2], mf_c[3]), (mm_c[4], mm_c[5]))
    draw_arrow(ax, (pf_c[2], pf_c[3]), (pc_c[4], pc_c[5]))
    draw_arrow(ax, (tf_c[2], tf_c[3]), (ts_c[4], ts_c[5]))
    draw_arrow(ax, (rf_c[2], rf_c[3]), (rt_c[4], rt_c[5]))
    draw_arrow(ax, (ef_c[2], ef_c[3]), (te_c[4], te_c[5]))

    plt.title("LightMem Class Architecture", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def generate_mermaid_files():
    project_structure_mmd = """graph TD
    subgraph LightMem["src/lightmem"]
        Configs["configs<br/>(Configuration)"]
        Factory["factory<br/>(Component Factory)"]
        Memory["memory<br/>(Core Logic)"]
        Toolkits["memory_toolkits<br/>(Utils & Baselines)"]
    end

    Memory -->|Depends| Configs
    Memory -->|Calls| Factory
    Factory -->|Instantiates| Components[Concrete Implementations<br/>(Manager, Retriever, Embedder...)]
    
    Configs -->|Defines| BaseConfigs[BaseMemoryConfigs]
    
    Toolkits -->|Extends| InferenceUtils[Inference Utils]
    Toolkits -->|Extends| Memories[Memories<br/>(Baselines: Mem0, LangMem)]
"""
    
    class_diagram_mmd = """classDiagram
    class LightMemory {
        +BaseMemoryConfigs config
        +PreCompressor compressor
        +TopicSegmenter segmenter
        +MemoryManager manager
        +TextEmbedder text_embedder
        +ContextRetriever context_retriever
        +EmbeddingRetriever embedding_retriever
        +GraphMem graph
        +add_memory()
        +retrieve()
    }

    class MemoryManagerFactory {
        +from_config()
    }
    class PreCompressorFactory {
        +from_config()
    }
    class TopicSegmenterFactory {
        +from_config()
    }
    class TextEmbedderFactory {
        +from_config()
    }
    class RetrieverFactory {
        +from_config()
    }

    LightMemory ..> MemoryManagerFactory : Use
    LightMemory ..> PreCompressorFactory : Use
    LightMemory ..> TopicSegmenterFactory : Use
    LightMemory ..> TextEmbedderFactory : Use
    LightMemory ..> RetrieverFactory : Use

    namespace Implementations {
        class MemoryManager {
            OpenAI
            DeepSeek
            LocalHFLoRA
        }
        class PreCompressor {
            EntropyCompress
            LLMLingua2
        }
        class TopicSegmenter {
            LLMLingua2
        }
        class TextEmbedder {
            HuggingFace
        }
        class Retriever {
            BM25 (Context)
            Qdrant (Embedding)
        }
    }

    MemoryManagerFactory ..> MemoryManager : Creates
    PreCompressorFactory ..> PreCompressor : Creates
    TopicSegmenterFactory ..> TopicSegmenter : Creates
    TextEmbedderFactory ..> TextEmbedder : Creates
    RetrieverFactory ..> Retriever : Creates
"""

    with open("project_structure.mmd", "w", encoding="utf-8") as f:
        f.write(project_structure_mmd)
    print("Saved project_structure.mmd")

    with open("class_diagram.mmd", "w", encoding="utf-8") as f:
        f.write(class_diagram_mmd)
    print("Saved class_diagram.mmd")

if __name__ == "__main__":
    generate_project_structure_img("project_structure.png")
    generate_class_diagram_img("class_diagram.png")
    generate_mermaid_files()
