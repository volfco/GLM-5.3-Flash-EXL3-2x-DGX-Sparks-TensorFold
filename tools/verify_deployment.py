#!/usr/bin/env python3
"""Exercise a live TensorFold deployment; write measured results as JSON."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', default='http://127.0.0.1:8000')
parser.add_argument('--output', default='logs/live-verification.json')
args = parser.parse_args()


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(args.base + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=900) as response:
        return json.load(response)


model = request('/v1/models')['data'][0]['id']
results = {'model': model, 'health_before': request('/health'), 'requests': []}


def completion(label, messages, draft=True, max_tokens=128):
    start = time.monotonic()
    reply = request('/v1/chat/completions', {
        'model': model, 'messages': messages, 'max_tokens': max_tokens,
        'temperature': 0, 'draft': draft,
        'chat_template_kwargs': {'enable_thinking': False},
    })
    record = {'label': label, 'elapsed_s': time.monotonic() - start,
              'usage': reply.get('usage'), 'finish_reason': reply['choices'][0]['finish_reason'],
              'content': reply['choices'][0]['message']['content']}
    assert record['content'], record
    return record


try:
    short = [{'role': 'user', 'content': 'Compute 17 * 19. Reply only with the integer.'}]
    for draft in (False, True):
        results['requests'].append(completion('draft=' + str(draft), short, draft))
    assert all(r['content'].strip() == '323' for r in results['requests']), results['requests']
    # More than the default spill threshold; unique records make a meaningful reusable prefix.
    system = '\n'.join(f'Record {i}: amber birch cedar delta elm fir ginkgo hazel.' for i in range(1200))
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': 'What word follows cedar in record 1199? Reply with one word.'}]
    cold = completion('cold-prefix', messages)
    results['requests'].append(cold)
    assert cold['content'].strip().lower().strip('.!') == 'delta', cold
    results['health_after_cold'] = request('/health')
    followup = messages + [{'role': 'assistant', 'content': cold['content']},
                           {'role': 'user', 'content': 'What word follows elm in record 1198? Reply with one word.'}]
    continued = completion('continued-prefix', followup)
    results['requests'].append(continued)
    assert continued['content'].strip().lower().strip('.!') == 'fir', continued
    assert ((continued.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens', 0) > 0, continued
    results['health_after_continuation'] = request('/health')
    def concurrent(i):
        return completion(f'concurrent-{i}', [{'role': 'user', 'content':
            f'Write a Python function named clamp_{i} that clamps a number to an inclusive lower and upper bound. Include a short explanation.'}])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results['requests'].extend(pool.map(concurrent, range(4)))
    results['health_after'] = request('/health')
    results['passed'] = True
finally:
    with open(args.output, 'w') as output:
        json.dump(results, output, indent=2)
    print(json.dumps({'passed': results.get('passed', False), 'output': args.output,
                      'requests': [{k: r[k] for k in ('label', 'elapsed_s', 'usage')} for r in results['requests']]}, indent=2))
