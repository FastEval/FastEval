import { createConversationItemE } from '../components/conversation-item.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { computeUpdatedHash } from '../utils.js'

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
    const dataById = {}
    for (const item of data.replies) {
        if (!(item.task_id in dataById))
            dataById[item.task_id] = []
        dataById[item.task_id].push(item)
    }

    for (const [itemId, items] of Object.entries(dataById)) {
        if (parameters.has('sample') && parameters.get('sample') !== itemId)
            continue

        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        samplesE.appendChild(itemE)

        function replace(completionNumber) {
            if (parameters.has('sample') && (!parameters.has('completion') || parseInt(parameters.get('completion')) !== (completionNumber + 1)))
                location = '#' + computeUpdatedHash({ completion: Math.min(Math.max(1, completionNumber + 1), items.length) })

            completionNumber = Math.min(Math.max(0, completionNumber), items.length - 1)

            const switchCompletionE = document.createElement('div')
            switchCompletionE.classList.add('switch-completion')

            const currentCompletionE = document.createElement('span')
            switchCompletionE.appendChild(currentCompletionE)
            currentCompletionE.textContent = ((completionNumber + 1) + '/' + items.length)

            const decreaseCompletionNumberE = document.createElement('button')
            decreaseCompletionNumberE.textContent = '<'
            switchCompletionE.prepend(decreaseCompletionNumberE)

            decreaseCompletionNumberE.addEventListener('click', () => { replace(completionNumber - 1) })

            const increaseCompletionNumberE = document.createElement('button')
            increaseCompletionNumberE.textContent = '>'
            switchCompletionE.appendChild(increaseCompletionNumberE)

            increaseCompletionNumberE.addEventListener('click', () => { replace(completionNumber + 1) })

            itemE.replaceChildren(
                createLinkE('ID: ' + itemId, { sample: itemId }),
                switchCompletionE,
                createTextE('The model was supposed to complete the following code:'),
                createConversationItemE('user', items[completionNumber].prompt),
                createTextE('The model gave the following code as output:'),
                createConversationItemE('assistant', items[completionNumber].completion_raw.trim()),
                createTextE('The following code was extracted:'),
                createConversationItemE('assistant', items[completionNumber].completion_processed.trim()),
                createTextE('This code ' + (items[completionNumber].success ? 'passed all' : 'failed some') + ' of the tests.'),
            )
        }

        if (parameters.has('completion'))
            replace(parseInt(parameters.get('completion')) - 1)
        else
            replace(0)
    }

    return containerE
}
