"""集中配置。"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cache_index_path(directory, pdf_path):
    # 索引位置
    # 1. 沿用绝对路径摘要作为文档标识，每份文档独立保存缓存。
    identity = hashlib.sha256(str(Path(pdf_path).resolve()).casefold().encode('utf-8')).hexdigest()[:24]
    return Path(directory).resolve() / identity / 'index.bin'


DEFAULT_MAX_PAGES = 30
DEFAULT_PDF = Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
DEFAULT_INDEX = cache_index_path(PROJECT_ROOT / 'data' / 'cache', DEFAULT_PDF)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "deepseek-flash"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_TIMEOUT = 60.0
LLM_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 500
TITLE_MAX_TOKENS = 64
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
RRF_C = 60
DENSE_WEIGHT = 0.8
SPARSE_WEIGHT = 1.2
CANDIDATE_K = 20
FINAL_K = 5
SEARCH_EF = 50
INDEX_M = 16
INDEX_SPACE = "cosine"


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    pdf_path: Path
    index_path: Path
    max_pages: int | None = DEFAULT_MAX_PAGES
    force_rebuild: bool = False
    embedding_model: str = EMBEDDING_MODEL

    @property
    def conversation_dir(self):
        # 会话目录
        # 1. 在项目的数据目录中定位聊天记录。
        return self.base_dir / "data" / "conversations"

    @property
    def document_dir(self):
        # 上传目录
        # 1. 在项目的数据目录中定位上传文件。
        return self.base_dir / "data" / "uploads"

    @property
    def index_dir(self):
        # 索引目录
        # 1. 在项目的数据目录中定位文档索引。
        return self.base_dir / "data" / "cache"

    @property
    def knowledge_settings_path(self):
        # 知识库设置
        # 1. 在数据目录中定位所有会话共用的文档设置。
        return self.base_dir / "data" / "settings" / "knowledge_base.json"

    @property
    def path_map_path(self):
        return self.base_dir / 'data' / 'settings' / 'path_map.json'

    @property
    def output_path(self):
        return self.index_path.with_suffix('.json')

    @property
    def evaluation_path(self):
        # 评测题集
        # 1. 相对项目目录找到评测问题。
        return self.base_dir / "tests" / "eval_questions.json"


def load_settings(*, base_dir=None, pdf_path=None, index_path=None,
                  max_pages=DEFAULT_MAX_PAGES, force_rebuild=False,
                  embedding_model=EMBEDDING_MODEL):
    # 加载配置
    # 1. 使用调用方给的设置，没给的部分使用原项目默认值。
    # 2. 将目录转成绝对路径，返回只读配置。
    root = Path(base_dir or PROJECT_ROOT).resolve()
    pdf = Path(pdf_path or DEFAULT_PDF).resolve()
    return Settings(
        base_dir=root,
        pdf_path=pdf,
        index_path=Path(index_path or cache_index_path(root / 'data' / 'cache', pdf)).resolve(),
        max_pages=max_pages,
        force_rebuild=force_rebuild,
        embedding_model=embedding_model,
    )
