"""离线回归检查：真实小型索引与语料，不下载模型、不调用大模型。

运行：python -B -m unittest discover -s test -p test_regressions.py -v
测试产物保留在系统临时目录，每次仅产生数 KB，不执行目录删除。
"""
import ast
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TinyModel:
    """用可预测的二维向量代替需要下载的嵌入模型。"""

    def encode(self, text):
        """让 apple 与 banana 对应两个不同方向。"""
        return np.array([1., 0.] if 'apple' in text else [0., 1.], dtype=np.float32)


class RegressionTests(unittest.TestCase):
    """覆盖首次建库、短语料检索、评测参数及分词边界。"""

    @classmethod
    def setUpClass(cls):
        """仅替换外部模型入口；本地 BM25、HNSW 与文件读写使用真实实现。"""
        embeddings = types.ModuleType('sentence_transformers')
        embeddings.SentenceTransformer = TinyModel
        llm = types.ModuleType('LLM')
        llm.ask_llm = lambda prompt: 'offline'
        # 保持替身到整组测试结束，避免恢复 sys.modules 后破坏 pickle 的类身份。
        modules = patch.dict(sys.modules, {'sentence_transformers': embeddings, 'LLM': llm})
        modules.start()
        cls.addClassCleanup(modules.stop)
        cls.dense = importlib.import_module('dense')
        cls.sparse = importlib.import_module('sparse')
        cls.service = importlib.import_module('rag_service')
        cls.evaluation = importlib.import_module('eval')

    def setUp(self):
        """为每个用例创建独立目录，避免触碰真实 PDF 和知识库。"""
        self.directory = Path(tempfile.mkdtemp(prefix='v6_regression_'))
        self.pdf = self.directory / 'sample.pdf'
        self.output = self.pdf.with_suffix('.json')
        self.index = self.directory / 'sample.bin'
        self.rows = [
            {'text': 'apple', 'file_name': 'sample.pdf', 'page_number': 1},
            {'text': 'banana', 'file_name': 'sample.pdf', 'page_number': 2},
        ]

    def test_first_build_creates_loadable_bm25(self):
        """首次建库应生成可读取的 BM25，且再次调用能复用缓存。"""
        self.create_pdf(['apple', 'banana'])
        with patch.object(self.service, 'get_pdf_text', wraps=self.service.get_pdf_text) as extract:
            self.service.building_data(self.pdf, str(self.index), TinyModel())
            self.service.building_data(self.pdf, str(self.index), TinyModel())
        extract.assert_called_once()
        self.assertEqual(self.sparse.load_bm25(self.pdf).corpus_size, 2)
        self.assertTrue(self.index.exists())

    def create_pdf(self, texts, path=None):
        """创建真实小型 PDF；空字符串页用于验证空白页也参与计数。"""
        with pymupdf.open() as document:
            for text in texts:
                page = document.new_page()
                if text:
                    page.insert_text((72, 72), text)
            document.save(path or self.pdf)

    def build(self, max_pages=11, force_rebuild=False, road=None, index=None):
        """通过生产入口建库，返回可观察的建库状态。"""
        return self.service.building_data(
            road or self.pdf, index or self.index, TinyModel(),
            max_pages=max_pages, force_rebuild=force_rebuild)

    def assert_cache_size(self, expected):
        """同时核对 JSON、向量、HNSW、BM25 的块数，防止只重建某一路。"""
        self.assertEqual(len(json.loads(self.output.read_text(encoding='utf-8'))), expected)
        self.assertEqual(np.load(self.pdf.with_suffix('.npy')).shape[0], expected)
        index = self.dense.hnswlib.Index(space='cosine', dim=2)
        index.load_index(str(self.index))
        self.assertEqual(index.get_current_count(), expected)
        self.assertEqual(self.sparse.load_bm25(self.pdf).corpus_size, expected)

    def test_page_increase_and_decrease_rebuild_all_caches(self):
        """实际读取页数增加或减少时，四种缓存一起更新。"""
        self.create_pdf(['apple', 'banana', 'pear'])
        for limit in (1, 3, 2):
            with self.subTest(limit=limit):
                result = self.build(limit)
                self.assertTrue(result['rebuilt'])
                self.assertEqual(result['actual_pages'], limit)
                self.assert_cache_size(limit)

    def test_equal_actual_pages_reuse_even_with_different_limits(self):
        """2 页 PDF 在上限 3、50、全文之间切换，均复用同一份缓存。"""
        self.create_pdf(['apple', 'banana'])
        self.build(3)
        with patch.object(self.service, 'get_pdf_text', side_effect=AssertionError('不应重新提取')):
            for limit in (3, 50, None):
                result = self.build(limit)
                self.assertFalse(result['rebuilt'])
                self.assertEqual(result['actual_pages'], 2)

    def test_blank_page_counts_as_processed(self):
        """末尾空白页没有文本块，但仍必须计入实际读取页数。"""
        self.create_pdf(['apple', ''])
        self.build(1)
        result = self.build(2)
        self.assertTrue(result['rebuilt'])
        self.assertEqual(result['actual_pages'], 2)
        self.assert_cache_size(1)
        self.assertFalse(self.build(2)['rebuilt'])

    def test_force_rebuild_refreshes_same_page_content(self):
        """同页数内容变化默认复用；强制重建时更新原文。"""
        self.create_pdf(['apple'])
        self.build()
        self.create_pdf(['banana'])
        self.assertFalse(self.build()['rebuilt'])
        self.assertTrue(self.build(force_rebuild=True)['rebuilt'])
        self.assertIn('banana', json.loads(self.output.read_text(encoding='utf-8'))[0]['text'])

    def test_legacy_cache_without_metadata_is_rebuilt(self):
        """没有建库记录的旧缓存不能因文件齐全而跳过首次迁移。"""
        self.create_pdf(['apple', 'banana'])
        self.output.write_text(json.dumps(self.rows), encoding='utf-8')
        self.dense.save_dense_index(self.output, self.index, TinyModel())
        self.sparse.save_bm25(self.sparse.build_bm25([['old']]), self.pdf)
        self.assertTrue(self.build()['rebuilt'])
        self.assert_cache_size(2)

    def test_missing_cache_rebuilds_without_deleting_files(self):
        """通过移动一个测试缓存模拟缺失，再验证整套缓存恢复。"""
        self.create_pdf(['apple', 'banana'])
        self.build()
        self.pdf.with_suffix('.pkl').rename(self.directory / 'saved_previous.pkl')
        self.assertTrue(self.build()['rebuilt'])
        self.assert_cache_size(2)

    def test_failed_force_rebuild_is_retried(self):
        """强制重建中途失败后，下次普通调用也必须重新完成建库。"""
        self.create_pdf(['apple', 'banana'])
        self.build()
        with patch.object(self.service, 'save_bm25', side_effect=RuntimeError('模拟写入失败')):
            with self.assertRaisesRegex(RuntimeError, '模拟写入失败'):
                self.build(force_rebuild=True)
        self.assertTrue(self.build()['rebuilt'])
        self.assertFalse(self.build()['rebuilt'])
        self.assert_cache_size(2)

    def test_shared_index_does_not_reuse_another_pdf(self):
        """相同页数但不同 PDF 复用同一索引路径时，必须重新关联。"""
        self.create_pdf(['apple'])
        other = self.directory / 'other.pdf'
        self.create_pdf(['banana'], other)
        self.build()
        self.assertTrue(self.build(road=other)['rebuilt'])
        self.assertTrue(self.build()['rebuilt'])

    def test_shared_pdf_caches_do_not_reuse_wrong_page_range(self):
        """同一 PDF 以不同页数写入另一个索引后，原索引的原文缓存已不匹配。"""
        self.create_pdf(['apple', 'banana'])
        self.build(1)
        self.build(2, index=self.directory / 'other.bin')
        self.assertTrue(self.build(1)['rebuilt'])
        self.assert_cache_size(1)

    def test_default_and_all_pages_extraction(self):
        """默认保持前 11 页；None 允许提取第 12 页及以后。"""
        self.create_pdf(['apple'] * 12)
        self.assertEqual(len(self.service.get_pdf_text(self.pdf)), 11)
        self.assertEqual(len(self.service.get_pdf_text(self.pdf, max_pages=None)), 12)

    def test_invalid_page_limit_is_rejected_before_writing(self):
        """页数参数必须为正整数或 None，非法值不产生缓存。"""
        self.create_pdf(['apple'])
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(value)
        self.assertFalse(self.output.exists())

    def test_corrupt_metadata_triggers_rebuild(self):
        """建库记录写入中断或被破坏时，恢复建库而不是复用不明状态。"""
        self.create_pdf(['apple'])
        self.build()
        self.pdf.with_suffix('.build.json').write_text('{', encoding='utf-8')
        self.assertTrue(self.build()['rebuilt'])
        self.assertFalse(self.build()['rebuilt'])

    def test_blank_only_pdf_reports_clear_error(self):
        """没有文本的选定范围不发布成功记录，并给出可理解的错误。"""
        self.create_pdf([''])
        with self.assertRaisesRegex(ValueError, '没有可提取的文本'):
            self.build()
        record = json.loads(self.pdf.with_suffix('.build.json').read_text(encoding='utf-8'))
        self.assertEqual(record['status'], 'building')
        self.assertFalse(self.index.exists())

    def test_answer_question_forwards_build_settings(self):
        """问答真实执行建库与融合检索，只有外部回答服务使用替身。"""
        self.create_pdf(['apple', 'banana'])
        with patch.object(self.service, 'building_data', wraps=self.service.building_data) as build:
            result = self.service.answer_question(
                self.pdf, str(self.index), 'apple', TinyModel(),
                max_pages=1, force_rebuild=True)
        self.assertEqual(build.call_args.kwargs['max_pages'], 1)
        self.assertTrue(build.call_args.kwargs['force_rebuild'])
        self.assertEqual(result['answer'], 'offline')
        self.assertEqual(len(result['sources']), 1)
        self.assertEqual(result['sources'][0]['page_number'], 1)

    def test_dense_request_larger_than_corpus(self):
        """请求 20 条时，只有两条的索引应返回两条，并保留正确排名。"""
        self.output.write_text(json.dumps(self.rows), encoding='utf-8')
        self.dense.save_dense_index(self.output, self.index, TinyModel())
        texts, scores, indices = self.dense.query_dense_text_top_k(
            'apple', str(self.index), self.output, TinyModel(), k=20)
        self.assertEqual(texts, ['apple', 'banana'])
        self.assertEqual(list(indices), [0, 1])
        self.assertEqual(len(scores), 2)

    def test_warm_question_reads_chunks_once(self):
        """一次缓存命中问答只解析一次文本 JSON，来源仍按原排名返回。"""
        self.create_pdf(['apple', 'banana'])
        self.build()
        with patch('json.load', wraps=json.load) as read:
            result = self.service.answer_question(self.pdf, str(self.index), 'apple', TinyModel())
        self.assertEqual([source['page_number'] for source in result['sources']], [1, 2])
        self.assertEqual(read.call_count, 1)

    def test_shared_data_preserves_retrieval_and_old_calls(self):
        """共享文本数据与旧的路径调用得到相同的下标、分数及文本。"""
        self.create_pdf(['apple', 'banana'])
        self.build()
        data = json.loads(self.output.read_text(encoding='utf-8'))
        query = importlib.import_module('query')
        model = TinyModel()
        bm25 = self.sparse.load_bm25(self.pdf)
        args = ('apple', str(self.index), self.output, model)
        old_dense = self.dense.query_dense_text_top_k(*args)
        old_sparse = self.sparse.query_bm25_top_k(self.output, bm25, 'apple')
        old_rrf = query.RRF_top_k_query_indece(*args, bm25)
        self.assertEqual(list(old_rrf[0]), [0, 1])
        self.assertEqual(old_rrf[1], [0.8 / 61, 0.8 / 62])
        with patch('json.load', side_effect=AssertionError('不应重复加载文本')):
            self.assertEqual(self.dense.query_dense_text_top_k(*args, data=data), old_dense)
            self.assertEqual(self.sparse.query_bm25_top_k(self.output, bm25, 'apple', data=data), old_sparse)
            self.assertEqual(query.RRF_top_k_query_indece(*args, bm25, data=data), old_rrf)
            self.assertEqual(query.top_k_query_indice2text([1, 0], self.output, data=data),
                             ['banana\n', 'apple\n'])
            self.assertEqual(query.top_k_query_indice2chunk([1, 0], self.output, data=data),
                             [data[1], data[0]])
            self.assertEqual(query.top_k_query_indice2chunk([], self.output, data=[]), [])

    def test_prompt_from_sources_preserves_exact_legacy_text(self):
        """提示词逐字符一致，继续支持原来的三个参数调用。"""
        prompt = importlib.import_module('prompt')
        self.output.write_text(json.dumps(self.rows), encoding='utf-8')
        expected = ('用户问题：apple\n以下是参考资料：[sample.pdf]\n'
                    '页码：2\n文本：banana\n\n[sample.pdf]\n页码：1\n文本：apple\n\n')
        self.assertEqual(prompt.get_prompt('apple', [1, 0], self.output), expected)
        self.assertEqual(prompt.format_prompt('apple', [self.rows[1], self.rows[0]]), expected)

    def test_shared_data_preserves_two_route_weighted_fusion(self):
        """两路均有命中且名次不同，融合权重、排名和得分保持原值。"""
        texts = ['apple', 'banana', 'banana banana', 'pear', 'apple apple']
        rows = [{'text': text, 'file_name': 'sample.pdf', 'page_number': i + 1}
                for i, text in enumerate(texts)]
        vectors = {'apple': [1., 0.], 'banana': [0., 1.], 'banana banana': [.3, 1.],
                   'pear': [.6, 1.], 'apple apple': [1., .2]}
        class Model:
            def encode(self, text):
                return np.array(vectors[text], dtype=np.float32)
        self.output.write_text(json.dumps(rows), encoding='utf-8')
        self.dense.save_dense_index(self.output, self.index, Model())
        bm25 = self.sparse.build_bm25([text.split() for text in texts])
        query = importlib.import_module('query')
        args = ('banana', str(self.index), self.output, Model(), bm25)
        expected_scores = [0.8 / 62 + 1.2 / 61, 0.8 / 61 + 1.2 / 62,
                           0.8 / 63, 0.8 / 64, 0.8 / 65]
        for options in ({}, {'data': rows}):
            indices, scores = query.RRF_top_k_query_indece(*args, **options)
            self.assertEqual(list(indices), [2, 1, 3, 4, 0])
            self.assertEqual(scores, expected_scores)

    def test_status_before_build_is_read_only_and_uses_actual_pages(self):
        """打开状态面板不触发建库；读取上限 30 的两页 PDF 显示实际两页。"""
        self.create_pdf(['apple', 'banana'])
        before = sorted(path.name for path in self.directory.iterdir())
        state = self.service.get_knowledge_base_status(self.pdf, self.index, max_pages=30)
        self.assertEqual(state['status'], 'not_built')
        self.assertEqual(state['actual_pages'], 2)
        self.assertEqual(sorted(path.name for path in self.directory.iterdir()), before)

    def test_status_checks_both_records_missing_caches_and_page_range(self):
        """状态与原建库规则一致：双记录配对、实际页数和四种缓存均需匹配。"""
        self.create_pdf(['apple', 'banana'])
        self.build(1)
        files = list(self.directory.iterdir())
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        state = self.service.get_knowledge_base_status(self.pdf, self.index, max_pages=1)
        self.assertEqual(state['status'], 'ready')
        for path, expected in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
        self.assertEqual(self.service.get_knowledge_base_status(
            self.pdf, self.index, max_pages=2)['status'], 'rebuild_required')
        index_record = self.index.with_suffix('.bin.meta.json')
        record = json.loads(index_record.read_text(encoding='utf-8'))
        record['pdf_path'] = str(self.directory / 'different.pdf')
        index_record.write_text(json.dumps(record), encoding='utf-8')
        self.assertEqual(self.service.get_knowledge_base_status(
            self.pdf, self.index, max_pages=1)['status'], 'rebuild_required')
        self.build(1)
        self.pdf.with_suffix('.pkl').rename(self.directory / 'previous.pkl')
        state = self.service.get_knowledge_base_status(self.pdf, self.index, max_pages=1)
        self.assertEqual(state['status'], 'rebuild_required')
        self.assertIn(str(self.pdf.with_suffix('.pkl')), state['missing_caches'])

    def test_status_distinguishes_unfinished_corrupt_and_unavailable(self):
        """遗留 building 记录表示未完成，不冒充正在后台构建或已就绪。"""
        self.create_pdf(['apple'])
        with patch.object(self.service, 'save_bm25', side_effect=RuntimeError('写入中断')):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertEqual(self.service.get_knowledge_base_status(
            self.pdf, self.index)['status'], 'incomplete')
        self.build()
        self.pdf.with_suffix('.build.json').write_text('{', encoding='utf-8')
        self.assertEqual(self.service.get_knowledge_base_status(
            self.pdf, self.index)['status'], 'rebuild_required')
        self.assertEqual(self.service.get_knowledge_base_status(
            self.directory / 'absent.pdf', self.index)['status'], 'unavailable')

    def test_optional_build_callback_keeps_answer_contract(self):
        """状态回调接收建库结果和已加载的块数，不增加回答字典字段或 JSON 读取。"""
        self.create_pdf(['apple', 'banana'])
        self.build()
        updates = []
        with patch('json.load', wraps=json.load) as read:
            result = self.service.answer_question(
                self.pdf, str(self.index), 'apple', TinyModel(), on_build_status=updates.append)
        self.assertEqual(set(result), {'answer', 'sources'})
        self.assertEqual(read.call_count, 1)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]['chunk_count'], 2)
        self.assertEqual(updates[0]['actual_pages'], 2)
        self.assertFalse(updates[0]['rebuilt'])
        self.assertEqual(updates[0]['cache_signature'],
                         self.service.get_knowledge_base_status(self.pdf, self.index)['cache_signature'])

    def test_eval_uses_final_k_without_global(self):
        """作为模块调用评估函数时不需要预先设置全局 k。"""
        self.check_evaluation()

    def test_eval_ignores_unrelated_global_k(self):
        """外部同名全局变量不应改变 Hit@k 的窗口。"""
        with patch.object(self.evaluation, 'k', 1, create=True):
            self.check_evaluation()

    def check_evaluation(self):
        """正确页位于第二名，final_k=2 时三种方法都应命中。"""
        self.output.write_text(json.dumps(self.rows), encoding='utf-8')
        with patch.object(self.evaluation, 'query_dense_text_top_k', return_value=([], [], [0, 1])), \
             patch.object(self.evaluation, 'query_bm25_top_k', return_value=([], [], [0, 1])), \
             patch.object(self.evaluation, 'RRF_top_k_query_indece', return_value=([0, 1], [])):
            result = self.evaluation.evaluate_model(
                [{'question': 'banana', 'gold_page': 2}], str(self.index),
                self.output, TinyModel(), None, final_k=2)
        for method in ('dense', 'sparse', 'rrf'):
            self.assertEqual(result[0][f'{method}_hit_at_k'], 1)
            self.assertEqual(result[0][f'{method}_hit_at_1'], 0)

    def test_eval_dataset_path_is_portable(self):
        """只执行入口里的路径赋值，确认评测集不依赖启动目录。"""
        tree = ast.parse((ROOT / 'eval.py').read_text(encoding='utf-8'))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'eval_set_path' for t in n.targets))
        scope = {'Path': Path, '__file__': str(ROOT / 'eval.py')}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'eval.py', 'exec'), scope)
        dataset = Path(scope['eval_set_path'])
        self.assertTrue(dataset.is_file())
        self.assertTrue(self.evaluation.load_eval_questions(dataset))

    def test_segment_removes_whitespace_and_preserves_words(self):
        """使用真实分词器，保留有意义词语并去掉空白、指定标点。"""
        tokens = self.sparse.segment_text('apple \n banana，')
        self.assertNotIn('', tokens)
        self.assertNotIn('，', tokens)
        self.assertIn('apple', tokens)
        self.assertIn('banana', tokens)


if __name__ == '__main__':
    unittest.main()
