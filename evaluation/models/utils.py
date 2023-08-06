def put_system_message_in_user_message(conversation):
    new_conversation = []
    current_user_message = None
    for item_type, item in conversation:
        if item_type == 'system':
            current_user_message = item
        elif item_type == 'assistant':
            if current_user_message is not None:
                raise
            new_conversation.append((item_type, item))
        elif item_type == 'user':
            if current_user_message is None:
                new_conversation.append((item_type, item))
            else:
                new_conversation.append((item_type, current_user_message + '\n\n' + item))
                current_user_message = None
        else:
            raise
    return new_conversation
