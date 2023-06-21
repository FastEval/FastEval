import { createConversationItemE } from '../components/conversation-item.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const explanationE = document.createElement('div')
    explanationE.classList.add('human-eval-plus__explanation')
    const informationLinkE = document.createElement('a')
    informationLinkE.textContent = 'here'
    informationLinkE.href = 'https://github.com/evalplus/evalplus'
    explanationE.append(
        createTextE('See '),
        informationLinkE,
        createTextE(' for more information on this benchmark.')
    )
    containerE.appendChild(explanationE)

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')

    const data = await (await fetch(baseUrl + '/human-eval-plus/' + parameters.get('model').replace('/', '--') + '.json')).json()
    for (const item of data.replies) {
        if (parameters.has('sample') && parameters.get('sample') !== item.task_id)
            continue

        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        samplesE.appendChild(itemE)
        itemE.append(
            createLinkE('ID: ' + item.task_id, { sample: item.task_id }),
            createTextE('The model was supposed to complete the following code:'),
            createConversationItemE('user', item.prompt),
            createTextE('The model gave the following code as output:'),
            createConversationItemE('assistant', item.completion_raw.trim()),
            createTextE('The following code was extracted:'),
            createConversationItemE('assistant', item.completion_processed.trim()),
            createTextE('This code ' + (item.success ? 'passed all' : 'failed some') + ' of the tests.'),
        )
    }

    return containerE
}
