import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.model_loader import EggieModel


def main():
    eggie = EggieModel()
    eggie.load_model()
    
    print("\n" + "="*50)
    print("Eggie 已启动！输入 'quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    conversation_history = []
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\nEggie: 拜拜啦！下次再聊～ 😊")
            break
        
        if not user_input:
            continue
        
        try:
            response = eggie.chat(user_input, conversation_history)
            print(f"\nEggie: {response}\n")
            
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"\n错误: {e}\n")


if __name__ == "__main__":
    main()