import itchat
from typing import Dict, List, Optional
from pathlib import Path
import sys
import json
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.model_loader import EggieModel


class WeChatBot:
    def __init__(self, eggie_model: EggieModel):
        self.eggie = eggie_model
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.max_history_length = 10
        
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        return self.conversation_history[user_id]
        
    def update_conversation_history(self, user_id: str, role: str, content: str):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            "role": role,
            "content": content
        })
        
        if len(self.conversation_history[user_id]) > self.max_history_length:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history_length:]
            
    def handle_text_message(self, msg):
        user_id = msg['FromUserName']
        user_message = msg['Text']
        
        try:
            conversation_history = self.get_conversation_history(user_id)
            
            response = self.eggie.chat(user_message, conversation_history)
            
            self.update_conversation_history(user_id, "user", user_message)
            self.update_conversation_history(user_id, "assistant", response)
            
            msg.reply(response)
            
            print(f"[微信] 用户 {msg['User']['NickName']}: {user_message}")
            print(f"[微信] Eggie: {response}")
            
        except Exception as e:
            error_msg = f"抱歉，我遇到了一些问题：{str(e)}"
            msg.reply(error_msg)
            print(f"[微信] 错误: {e}")
            
    def handle_friend_request(self, msg):
        try:
            itchat.add_friend(**msg['Text'])
            msg.verify()
            print(f"[微信] 已添加好友: {msg['RecommendInfo']['NickName']}")
        except Exception as e:
            print(f"[微信] 添加好友失败: {e}")
            
    def handle_group_message(self, msg):
        if msg.isAt:
            user_id = msg['FromUserName']
            user_message = msg['Text']
            
            try:
                conversation_history = self.get_conversation_history(user_id)
                
                response = self.eggie.chat(user_message, conversation_history)
                
                self.update_conversation_history(user_id, "user", user_message)
                self.update_conversation_history(user_id, "assistant", response)
                
                msg.reply(f"@{msg['ActualNickName']} {response}")
                
                print(f"[微信群] {msg['User']['NickName']}: {user_message}")
                print(f"[微信群] Eggie: {response}")
                
            except Exception as e:
                error_msg = f"抱歉，我遇到了一些问题：{str(e)}"
                msg.reply(error_msg)
                print(f"[微信群] 错误: {e}")
                
    def start(self, auto_reply: bool = True):
        print("正在启动微信机器人...")
        
        itchat.auto_login(hotReload=True)
        
        if auto_reply:
            itchat.msg_register(itchat.content.TEXT, isFriendChat=True)(self.handle_text_message)
            itchat.msg_register(itchat.content.TEXT, isGroupChat=True)(self.handle_group_message)
            itchat.msg_register(itchat.content.FRIENDS)(self.handle_friend_request)
        
        print("微信机器人已启动！")
        print("提示：在微信中发送消息给Eggie即可开始对话")
        print("在群聊中@Eggie可以触发回复")
        print("按 Ctrl+C 退出")
        
        itchat.run()
        
    def send_message(self, to_user_name: str, message: str):
        try:
            itchat.send(message, toUserName=to_user_name)
            print(f"[微信] 发送消息: {message}")
            return True
        except Exception as e:
            print(f"[微信] 发送消息失败: {e}")
            return False
            
    def get_friends(self):
        friends = itchat.get_friends(update=True)
        return friends
        
    def get_chatrooms(self):
        chatrooms = itchat.get_chatrooms(update=True)
        return chatrooms


def main():
    eggie = EggieModel()
    eggie.load_model()
    
    bot = WeChatBot(eggie)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\n正在退出微信机器人...")
        itchat.logout()
        print("已退出")


if __name__ == "__main__":
    main()
