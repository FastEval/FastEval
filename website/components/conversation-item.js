export function createConversationItemE(role, text) {
    const containerE = document.createElement('div')
    containerE.classList.add('conversation-item')
    containerE.classList.add('conversation-item-' + role)

    const roleE = document.createElement('span')
    roleE.classList.add('conversation-item__role')
    roleE.textContent = role.charAt(0).toUpperCase() + role.slice(1)
    containerE.appendChild(roleE)

    const contentE = document.createElement('pre')
    contentE.textContent = text
    containerE.appendChild(contentE)

    return containerE
}
