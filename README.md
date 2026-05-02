```markdown
# Calculator API

This is a simple calculator API built with Flask. It provides endpoints for performing basic arithmetic operations (addition, subtraction, multiplication, and division) and resetting the calculator.

## Endpoints

### `/calculate` (POST)

Performs an arithmetic operation.

**Request Body:**
```json
{
  "operation": "add|subtract|multiply|divide",
  "number1": 5.0,
  "number2": 3.0
}
```

**Response:**
- Success: `{ "result": <calculated_result> }`
- Error: `{ "error": <error_message> }`

### `/reset` (POST)

Resets the calculator.

**Response:**
`{ "message": "Calculator reset" }`

## Usage

1. Clone the repository: `git clone https://github.com/mannaandpoem/simple_calculator.git`
2. Install dependencies: `pip install flask`
3. Run the application: `python main.py`
4. The API will be running at `http://localhost:5000`

vllm serve Qwen/Qwen3.5-0.8B --port 8000 --tensor-parallel-size 1 --max-model-len 262144
