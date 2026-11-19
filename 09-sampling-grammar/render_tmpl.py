"""从 GGUF 里取出 tokenizer.chat_template，用 jinja2 渲染几种消息，看模板到底拼出了什么。

llama-completion 不把 --reasoning、--chat-template-kwargs 传给模板（0ec4027），
想看 enable_thinking 的效果只能自己渲染。server 会把这些参数传进去。

用法：python3 render_tmpl.py MODEL.gguf [--gguf-py 路径]
"""
import argparse
import json
import sys


def load_template(path, gguf_py):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    r = GGUFReader(path)
    f = r.fields['tokenizer.chat_template']
    return bytes(f.parts[f.data[0]]).decode()


def render(tmpl, messages, **kw):
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, loader=jinja2.BaseLoader())
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    env.filters['tojson'] = lambda x, **k: json.dumps(x, ensure_ascii=False)
    return env.from_string(tmpl).render(messages=messages, bos_token='', eos_token='<|im_end|>', **kw)


CASES = [
    ('单轮，缺省', [{'role': 'user', 'content': 'What is 2+3?'}], dict(add_generation_prompt=True)),
    ('单轮，enable_thinking=false', [{'role': 'user', 'content': 'What is 2+3?'}],
     dict(add_generation_prompt=True, enable_thinking=False)),
    ('带系统提示词', [{'role': 'system', 'content': 'Answer briefly.'}, {'role': 'user', 'content': 'What is 2+3?'}],
     dict(add_generation_prompt=True)),
    ('多轮：上一轮的思考被删掉', [
        {'role': 'user', 'content': 'What is 2+3?'},
        {'role': 'assistant', 'content': '<think>\n2 plus 3 is 5.\n</think>\n\n2 + 3 = 5.'},
        {'role': 'user', 'content': 'And 4+4?'}], dict(add_generation_prompt=True)),
    ('带工具', [{'role': 'user', 'content': 'Weather in Paris?'}],
     dict(add_generation_prompt=True, tools=[{'type': 'function', 'function': {
         'name': 'get_weather', 'description': 'Get weather', 'parameters': {
             'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--gguf-py', default='')
    a = ap.parse_args()
    t = load_template(a.model, a.gguf_py)
    print(f'# chat_template {len(t)} 字符')
    for name, msgs, kw in CASES:
        out = render(t, msgs, **kw)
        print(f'== {name}（{len(out)} 字符）')
        print(out.replace('\n', '\\n\n'), end='')
        print('\n')


if __name__ == '__main__':
    main()
