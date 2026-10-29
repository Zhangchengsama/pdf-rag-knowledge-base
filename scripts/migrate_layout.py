"""一次性迁移运行数据；逐文件校验并移动，保留原件，冲突时停止。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
import cache_store
from config import cache_index_path
from document_settings import normalize_document


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def migration_actions(root):
    # 迁移计划
    # 1. 收集旧设置、上传 PDF 和旧请求中引用的文档。
    # 2. 为可复用缓存生成复制计划，为原始资料和聊天生成移动计划。
    # 3. 设置原件单独保留，聊天正文及元数据逐字节保留。
    old_data = root / 'v6/data'
    old_settings = old_data / 'knowledge_base.json'
    documents, active = {}, None

    def remember(document):
        if document:
            document = normalize_document(document)
            documents.setdefault(document['pdf_path'], document)
            return document

    if old_settings.exists():
        active = remember(read_json(old_settings))
    for metadata in (old_data / 'conversations').glob('*.meta.json'):
        value = read_json(metadata)
        remember(value.get('document'))
        remember((value.get('request') or {}).get('document'))
    for pdf in (old_data / 'documents').rglob('*.pdf'):
        key = hashlib.sha256(str(pdf.resolve()).casefold().encode('utf-8')).hexdigest()[:24]
        remember({'pdf_path': str(pdf), 'index_path': str(old_data / 'indexes' / f'{key}.bin'),
                  'max_pages': 30})

    actions, mapping = [], {}

    def transfer(source, target, kind):
        source, target = Path(source).resolve(), Path(target).resolve()
        if source.is_file():
            actions.append({'kind': kind, 'source': str(source), 'target': str(target),
                            'sha256': checksum(source), 'bytes': source.stat().st_size})

    def write(target, value):
        content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        actions.append({'kind': 'write', 'target': str(target.resolve()), 'text': content,
                        'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()})

    for old_pdf, document in documents.items():
        pdf = Path(old_pdf)
        new_pdf = (root / 'data/uploads' / pdf.relative_to(old_data / 'documents')
                   if pdf.is_relative_to(old_data / 'documents') else pdf)
        index = cache_index_path(root / 'data/cache', new_pdf)
        target = {'pdf_path': str(new_pdf), 'index_path': str(index),
                  'output_path': str(index.with_suffix('.json'))}
        mapping[old_pdf] = target
        paths = cache_store.cache_paths(pdf, document['index_path'], document.get('output_path'))
        records = cache_store.read_build_records(cache_store.metadata_paths(
            pdf, document['index_path'], document.get('output_path')))
        record = records[0]
        reusable = (isinstance(record, dict) and records[0] == records[1]
                    and record.get('status') == 'ready' and record.get('pdf_path') == old_pdf
                    and record.get('index_path') == document['index_path']
                    and all(path.is_file() for path in paths))
        if reusable:
            new_paths = cache_store.cache_paths(new_pdf, index, target['output_path'])
            for source, destination in zip(paths, new_paths):
                transfer(source, destination, 'copy')
            translated = dict(record, pdf_path=str(new_pdf), index_path=str(index))
            for destination in cache_store.metadata_paths(new_pdf, index, target['output_path']):
                write(destination, translated)

    for source in old_data.rglob('*'):
        if not source.is_file():
            continue
        relative = source.relative_to(old_data)
        group = relative.parts[0]
        if source == old_settings:
            destination = root / 'data/settings/legacy_knowledge_base.json'
        elif group == 'conversations':
            destination = root / 'data' / relative
        elif group == 'documents' and source.suffix.lower() == '.pdf':
            destination = root / 'data/uploads' / Path(*relative.parts[1:])
        else:
            destination = root / 'data/cache/legacy' / relative
        transfer(source, destination, 'move')
    write(root / 'data/settings/path_map.json', mapping)
    if active:
        write(root / 'data/settings/knowledge_base.json',
              dict(mapping[active['pdf_path']], max_pages=active['max_pages']))
    return actions


def migrate(project_root=PROJECT):
    # 执行迁移
    # 1. 校验全部目标冲突及移动路径边界，之后才修改文件。
    # 2. 保存可恢复计划；逐文件复制、移动或写入，并校验 SHA256。
    # 3. 保存报告，重复执行已完成的迁移不会改写文件。
    root = Path(project_root).resolve()
    settings = root / 'data/settings'
    report_path, plan_path = settings / 'migration_report.json', settings / 'migration_plan.json'
    if report_path.exists():
        return read_json(report_path)
    actions = read_json(plan_path) if plan_path.exists() else migration_actions(root)
    destinations = set()
    for action in actions:
        target = Path(action['target']).resolve()
        if not target.is_relative_to(root) or str(target) in destinations:
            raise ValueError(f'迁移目标不合法或重复：{target}')
        destinations.add(str(target))
        if action['kind'] == 'move' and not Path(action['source']).resolve().is_relative_to(root):
            raise ValueError(f'禁止移动项目外文件：{action["source"]}')
        if target.exists() and (not target.is_file() or checksum(target) != action['sha256']):
            raise FileExistsError(f'目标已有不同内容，迁移未开始：{target}')
        if action['kind'] != 'write':
            source = Path(action['source'])
            if source.exists() and checksum(source) != action['sha256']:
                raise ValueError(f'原文件自计划生成后已变化：{source}')
            if not source.exists() and not target.exists():
                raise FileNotFoundError(source)
    settings.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        plan_path.write_text(json.dumps(actions, ensure_ascii=False, indent=2), encoding='utf-8')
    for action in actions:
        target = Path(action['target'])
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if action['kind'] == 'move':
            shutil.move(action['source'], str(target))
        elif action['kind'] == 'copy':
            shutil.copy2(action['source'], target)
        else:
            target.write_bytes(action['text'].encode('utf-8'))
        if checksum(target) != action['sha256']:
            raise ValueError(f'迁移后校验失败：{target}')
    report = {'actions': actions, 'verified_files': len(actions),
              'copied_bytes': sum(item.get('bytes', 0) for item in actions if item['kind'] == 'copy')}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    options = parser.parse_args()
    result = migrate(options.root)
    print(f"已校验 {result['verified_files']} 项；新增缓存副本 {result['copied_bytes']} 字节。")
