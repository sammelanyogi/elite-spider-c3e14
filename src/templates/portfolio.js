import React from 'react';
import _ from 'lodash';
import {graphql} from 'gatsby';

import components, {Layout} from '../components/index';
import {getPages, Link, withPrefix} from '../utils';

// this minimal GraphQL query ensures that when 'gatsby develop' is running,
// any changes to content files are reflected in browser
export const query = graphql`
  query($url: String) {
    sitePage(path: {eq: $url}) {
      id
    }
  }
`;

export default class Portfolio extends React.Component {
    render() {
        let display_projects = _.orderBy(getPages(this.props.pageContext.pages, '/portfolio'), 'frontmatter.date', 'desc');
        return (
            <Layout {...this.props}>
            <div className="inner outer">
              <header className="page-header inner-sm">
                <h1 className="page-title line-top">{_.get(this.props, 'pageContext.frontmatter.title', null)}</h1>
                {_.get(this.props, 'pageContext.frontmatter.subtitle', null) && (
                <p className="page-subtitle">{_.get(this.props, 'pageContext.frontmatter.subtitle', null)}</p>
                )}
              </header>
              {_.map(_.get(this.props, 'pageContext.frontmatter.sections', null), (section, section_idx) => {
                  let component = _.upperFirst(_.camelCase(_.get(section, 'type', null)));
                  let Component = components[component];
                  return (
                    <Component key={section_idx} {...this.props} section={section} site={this.props.pageContext.site} />
                  )
              })}
              <div className={'portfolio-feed layout-' + _.get(this.props, 'pageContext.frontmatter.layout_style', null)}>
                {_.map(display_projects, (post, post_idx) => (
                <article key={post_idx} className="project">
                  <Link to={withPrefix(_.get(post, 'url', null))} className="project-link">
                    {_.get(post, 'frontmatter.thumb_image', null) && (
                    <div className="project-thumbnail">
                      <img src={withPrefix(_.get(post, 'frontmatter.thumb_image', null))} alt={_.get(post, 'frontmatter.title', null)} />
                    </div>
                    )}
                    <header className="project-header">
                      <h2 className="project-title">{_.get(post, 'frontmatter.title', null)}</h2>
                    </header>
                  </Link>
                </article>
                ))}
              </div>
            </div>
            </Layout>
        );
    }
}
