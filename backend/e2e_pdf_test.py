import json
import pathlib
import sys

import httpx

PDF_PATH = pathlib.Path('/Users/yash/Downloads/IOT Unit-2 @zammers.pdf')
BASE_URL = 'http://localhost:8000/api/v1'


def main() -> int:
    if not PDF_PATH.exists():
        print(f'ERROR: PDF not found at {PDF_PATH}')
        return 1

    with httpx.Client(timeout=180.0) as client:
        with PDF_PATH.open('rb') as f:
            files = {'file': (PDF_PATH.name, f, 'application/pdf')}
            upload = client.post(f'{BASE_URL}/documents/upload', files=files)

        print('UPLOAD_STATUS', upload.status_code)
        if upload.status_code != 200:
            print(upload.text)
            return 1

        upload_json = upload.json()
        doc_id = upload_json['doc_id']
        print('DOC_ID', doc_id)
        print('UPLOAD_RESULT', json.dumps(upload_json, indent=2))

        chunk_payload = {
            'doc_id': doc_id,
            'question': 'Summarize the key topics in this unit.',
            'top_k': 3,
        }
        chunks_resp = client.post(f'{BASE_URL}/chunks/test', json=chunk_payload)
        print('CHUNKS_STATUS', chunks_resp.status_code)
        if chunks_resp.status_code == 200:
            chunks_json = chunks_resp.json()
            print('CHUNKS_RETURNED', chunks_json.get('returned_chunks'))
            for i, chunk in enumerate(chunks_json.get('chunks', []), 1):
                snippet = chunk.get('content', '').replace('\n', ' ')[:220]
                print(f'CHUNK_{i}_SCORE', chunk.get('score'))
                print(f'CHUNK_{i}_SNIPPET', snippet)
        else:
            print(chunks_resp.text)

        query_payload = {
            'doc_id': doc_id,
            'question': 'What are the main points from this IoT Unit 2 document?',
        }
        query_resp = client.post(f'{BASE_URL}/query', json=query_payload)
        print('QUERY_STATUS', query_resp.status_code)
        if query_resp.status_code == 200:
            query_json = query_resp.json()
            print('ANSWER_PREVIEW', query_json.get('answer', '')[:700])
            print('SOURCES_COUNT', len(query_json.get('sources', [])))
            print('RESPONSE_TIME_MS', query_json.get('response_time_ms'))
            return 0

        print(query_resp.text)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
