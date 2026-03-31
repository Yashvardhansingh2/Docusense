import httpx

DOC_ID = '3248cc5d-fbf0-4dde-b177-c380e2b461f9'
BASE_URL = 'http://localhost:8000/api/v1/query'

QUESTIONS = [
    'What is this PDF about?',
    'List the main topics covered in Unit 1.',
    'What does the document say about IoT safety and security?',
    'Explain the Publish-Subscribe model described in the document.',
    'What are the key characteristics of IoT mentioned in this unit?',
]


def main() -> None:
    with httpx.Client(timeout=180.0) as client:
        for i, question in enumerate(QUESTIONS, 1):
            response = client.post(BASE_URL, json={'doc_id': DOC_ID, 'question': question})
            print(f'Q{i}: {question}')
            print('STATUS:', response.status_code)
            if response.status_code == 200:
                data = response.json()
                print('ANSWER:', data.get('answer', '').strip())
                print('SOURCES:', data.get('sources'))
                print('TIME_MS:', data.get('response_time_ms'))
                print('FROM_CACHE:', data.get('from_cache'))
            else:
                print('ERROR:', response.text)
            print('-' * 80)


if __name__ == '__main__':
    main()
