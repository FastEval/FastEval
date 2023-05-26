def put_system_message_in_prompter_message(conversation):
    new_conversation = []
    current_prompter_message = None
    for item_type, item in conversation:
        if item_type == 'system':
            current_prompter_message = item
        elif item_type == 'assistant':
            if current_prompter_message is not None:
                raise
            new_conversation.append((item_type, item))
        elif item_type == 'user':
            if current_prompter_message is None:
                new_conversation.append((item_type, item))
            else:
                new_conversation.append((item_type, current_prompter_message + '\n\n' + item))
                current_prompter_message = None
        else:
            raise
    return new_conversation
