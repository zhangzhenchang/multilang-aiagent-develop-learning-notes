import { useState, useEffect } from 'react';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
  edited: boolean;
}

function App() {
  const [todos, setTodos] = useState<Todo[]>(() => {
    const saved = localStorage.getItem('todos');
    return saved ? JSON.parse(saved) : [];
  });
  const [newTodo, setNewTodo] = useState('');
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');

  // Save to localStorage whenever todos change
  useEffect(() => {
    localStorage.setItem('todos', JSON.stringify(todos));
  }, [todos]);

  const addTodo = () => {
    if (newTodo.trim() === '') return;
    const newTodoItem: Todo = {
      id: Date.now().toString(),
      text: newTodo.trim(),
      completed: false,
      edited: false,
    };
    setTodos([...todos, newTodoItem]);
    setNewTodo('');
  };

  const deleteTodo = (id: string) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  const toggleTodo = (id: string) => {
    setTodos(
      todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  const startEditing = (todo: Todo) => {
    setEditingId(todo.id);
    setEditText(todo.text);
  };

  const saveEdit = () => {
    if (editText.trim() === '') return;
    setTodos(
      todos.map(todo =>
        todo.id === editingId ? { ...todo, text: editText.trim(), edited: true } : todo
      )
    );
    setEditingId(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const filteredTodos = todos.filter(todo => {
    if (filter === 'active') return !todo.completed;
    if (filter === 'completed') return todo.completed;
    return true;
  });

  const activeCount = todos.filter(todo => !todo.completed).length;
  const completedCount = todos.length - activeCount;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (editingId) {
        saveEdit();
      } else {
        addTodo();
      }
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>✨ Todo List</h1>
          <p>Organize your tasks with style</p>
        </header>

        <div className="input-section">
          <input
            type="text"
            value={newTodo}
            onChange={(e) => setNewTodo(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Add a new task..."
            className="todo-input"
          />
          <button onClick={addTodo} className="add-btn">
            ➕ Add
          </button>
        </div>

        <div className="stats">
          <span className="stat-item">
            <strong>{activeCount}</strong> {activeCount === 1 ? 'task' : 'tasks'} left
          </span>
          <span className="stat-item">
            <strong>{completedCount}</strong> completed
          </span>
        </div>

        <div className="filter-section">
          <button 
            onClick={() => setFilter('all')}
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          >
            All
          </button>
          <button 
            onClick={() => setFilter('active')}
            className={`filter-btn ${filter === 'active' ? 'active' : ''}`}
          >
            Active
          </button>
          <button 
            onClick={() => setFilter('completed')}
            className={`filter-btn ${filter === 'completed' ? 'active' : ''}`}
          >
            Completed
          </button>
        </div>

        <div className="todos-list">
          {filteredTodos.length === 0 ? (
            <div className="empty-state">
              <p>No tasks found. Add one above!</p>
            </div>
          ) : (
            filteredTodos.map((todo) => (
              <div 
                key={todo.id} 
                className={`todo-item ${todo.completed ? 'completed' : ''} ${editingId === todo.id ? 'editing' : ''}`}
                style={{ animation: 'fadeIn 0.3s ease-out' }}
              >
                {editingId === todo.id ? (
                  <div className="edit-form">
                    <input
                      type="text"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') saveEdit();
                        if (e.key === 'Escape') cancelEdit();
                      }}
                      autoFocus
                      className="edit-input"
                    />
                    <div className="edit-actions">
                      <button onClick={saveEdit} className="save-btn">✓</button>
                      <button onClick={cancelEdit} className="cancel-btn">✕</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="todo-content">
                      <input
                        type="checkbox"
                        checked={todo.completed}
                        onChange={() => toggleTodo(todo.id)}
                        className="todo-checkbox"
                      />
                      <span className="todo-text">{todo.text}</span>
                    </div>
                    <div className="todo-actions">
                      <button 
                        onClick={() => startEditing(todo)}
                        className="edit-btn"
                        aria-label="Edit todo"
                      >
                        ✏️
                      </button>
                      <button 
                        onClick={() => deleteTodo(todo.id)}
                        className="delete-btn"
                        aria-label="Delete todo"
                      >
                        🗑️
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))
          )}
        </div>

        {todos.length > 0 && (
          <div className="clear-section">
            <button 
              onClick={() => setTodos(todos.filter(todo => !todo.completed))}
              className="clear-btn"
            >
              Clear Completed
            </button>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .app {
          min-height: 100vh;
          background: linear-gradient(135deg, #4f46e5, #7c3aed, #ec4899);
          padding: 2rem 0;
        }
        
        .container {
          max-width: 600px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        .header {
          text-align: center;
          margin-bottom: 2rem;
        }
        
        .header h1 {
          color: white;
          font-size: 2.5rem;
          margin-bottom: 0.5rem;
          text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
          color: rgba(255,255,255,0.8);
          font-size: 1.1rem;
        }
        
        .input-section {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
        }
        
        .todo-input {
          flex: 1;
          padding: 0.75rem 1rem;
          border: none;
          border-radius: 8px;
          font-size: 1rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.15);
          transition: all 0.2s ease;
        }
        
        .todo-input:focus {
          outline: none;
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .add-btn {
          padding: 0.75rem 1.5rem;
          background: white;
          color: #4f46e5;
          border: none;
          border-radius: 8px;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.2s ease;
          box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .add-btn:hover {
          background: #f9fafb;
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .stats {
          display: flex;
          justify-content: space-between;
          margin-bottom: 1.5rem;
          color: white;
          font-size: 0.9rem;
        }
        
        .stat-item {
          background: rgba(255,255,255,0.15);
          padding: 0.5rem 1rem;
          border-radius: 20px;
          backdrop-filter: blur(10px);
        }
        
        .filter-section {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
        }
        
        .filter-btn {
          flex: 1;
          padding: 0.5rem;
          background: rgba(255,255,255,0.15);
          border: none;
          border-radius: 20px;
          color: white;
          cursor: pointer;
          transition: all 0.2s ease;
          backdrop-filter: blur(10px);
        }
        
        .filter-btn.active {
          background: white;
          color: #4f46e5;
          font-weight: bold;
        }
        
        .filter-btn:hover:not(.active) {
          background: rgba(255,255,255,0.25);
        }
        
        .todos-list {
          margin-bottom: 1.5rem;
        }
        
        .todo-item {
          background: white;
          border-radius: 12px;
          padding: 1rem;
          margin-bottom: 0.75rem;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          display: flex;
          justify-content: space-between;
          align-items: center;
          transition: all 0.3s ease;
          animation: fadeIn 0.3s ease-out;
        }
        
        .todo-item:hover:not(.editing) {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        .todo-item.completed {
          opacity: 0.7;
        }
        
        .todo-content {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
        
        .todo-checkbox {
          width: 1.5rem;
          height: 1.5rem;
          cursor: pointer;
        }
        
        .todo-text {
          font-size: 1.1rem;
          color: #374151;
        }
        
        .todo-item.completed .todo-text {
          text-decoration: line-through;
          color: #6b7280;
        }
        
        .todo-actions {
          display: flex;
          gap: 0.5rem;
        }
        
        .edit-btn, .delete-btn {
          background: none;
          border: none;
          width: 2.5rem;
          height: 2.5rem;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          font-size: 1.1rem;
          transition: all 0.2s ease;
        }
        
        .edit-btn:hover {
          background: #f9fafb;
        }
        
        .delete-btn:hover {
          background: #fef2f2;
          color: #ef4444;
        }
        
        .edit-form {
          display: flex;
          width: 100%;
          gap: 0.5rem;
        }
        
        .edit-input {
          flex: 1;
          padding: 0.5rem 1rem;
          border: 2px solid #4f46e5;
          border-radius: 8px;
          font-size: 1rem;
        }
        
        .edit-actions {
          display: flex;
          gap: 0.25rem;
        }
        
        .save-btn, .cancel-btn {
          width: 2.5rem;
          height: 2.5rem;
          border-radius: 50%;
          border: none;
          font-size: 1.2rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .save-btn {
          background: #4f46e5;
          color: white;
        }
        
        .save-btn:hover {
          background: #4338ca;
        }
        
        .cancel-btn {
          background: #f9fafb;
          color: #6b7280;
        }
        
        .cancel-btn:hover {
          background: #f4f4f5;
        }
        
        .empty-state {
          text-align: center;
          padding: 2rem;
          color: #6b7280;
        }
        
        .clear-section {
          text-align: center;
        }
        
        .clear-btn {
          padding: 0.5rem 1.5rem;
          background: rgba(255,255,255,0.15);
          color: white;
          border: none;
          border-radius: 20px;
          cursor: pointer;
          transition: all 0.2s ease;
          backdrop-filter: blur(10px);
        }
        
        .clear-btn:hover {
          background: rgba(255,255,255,0.25);
        }
        
        @media (max-width: 600px) {
          .input-section {
            flex-direction: column;
          }
          
          .filter-section {
            flex-wrap: wrap;
          }
          
          .stats {
            flex-direction: column;
            gap: 0.5rem;
          }
        }
      `}</style>
    </div>
  );
}

export default App
