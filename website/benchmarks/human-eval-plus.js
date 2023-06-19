import { createConversationItemE } from '../components/conversation-item.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')
    containerE.classList.add('samples')

    const data = await (await fetch(baseUrl + '/human-eval-plus/' + parameters.get('model').replace('/', '--') + '.json')).json()
    for (const item of data.replies) {
        if (parameters.has('sample') && parameters.get('sample') !== item.task_id)
            continue

        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        containerE.appendChild(itemE)
        itemE.append(
            createLinkE('ID: ' + item.task_id, { sample: item.task_id }),
            createTextE('The model was supposed to complete the following code:'),
            createConversationItemE('user', item.prompt),
            createTextE('The model gave the following code as output:'),
            createConversationItemE('assistant', item.completion_raw),
            createTextE('The following code was extracted:'),
            createConversationItemE('assistant', item.completion_processed),
            createTextE('This code ' + (item.success ? 'passed all' : 'failed some') + ' of the tests.'),
        )
    }

    return containerE
}
